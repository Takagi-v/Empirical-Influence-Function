from openai import OpenAI
import json
from src.annotate.utils import TokenCorrelation, SubwordToken
import importlib
import os
import random
import threading
import time
from pathlib import Path

_key_file = Path("./data/openai_key.txt")
if not os.environ.get("OPENAI_API_KEY") and _key_file.exists():
    os.environ["OPENAI_API_KEY"] = _key_file.read_text(encoding="utf-8").strip()

from dataclasses import dataclass

_OPENAI_RATE_LOCK = threading.Lock()
_OPENAI_LAST_REQUEST_TS = 0.0
_OPENAI_CLIENT_LOCAL = threading.local()


def get_thread_local_openai_client() -> OpenAI:
    """Reuse one OpenAI client per worker thread.

    Client construction is not the main bottleneck, but avoiding per-row client
    setup removes a small fixed overhead without changing annotation prompts or
    model behavior.
    """
    client = getattr(_OPENAI_CLIENT_LOCAL, "client", None)
    if client is None:
        client = OpenAI()
        _OPENAI_CLIENT_LOCAL.client = client
    return client


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _rate_limited_chat_completion(client: OpenAI, **kwargs):
    global _OPENAI_LAST_REQUEST_TS
    min_interval = float(os.environ.get("ANNOTATE_MIN_REQUEST_INTERVAL", "1.2"))
    max_retries = int(os.environ.get("ANNOTATE_MAX_RETRIES", "8"))
    base_sleep = float(os.environ.get("ANNOTATE_RETRY_BASE_SLEEP", "10"))
    request_timeout = os.environ.get("ANNOTATE_REQUEST_TIMEOUT")
    if request_timeout and "timeout" not in kwargs:
        kwargs["timeout"] = float(request_timeout)

    for attempt in range(max_retries + 1):
        with _OPENAI_RATE_LOCK:
            now = time.monotonic()
            wait = min_interval - (now - _OPENAI_LAST_REQUEST_TS)
            if wait > 0:
                time.sleep(wait)
            _OPENAI_LAST_REQUEST_TS = time.monotonic()
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt >= max_retries or not _is_rate_limit_error(exc):
                raise
            # ModelArts sometimes reports both 1 QPS and 3 RPM. Back off generously.
            sleep_s = max(base_sleep * (attempt + 1), min_interval) + random.uniform(0, 1.5)
            print(f"[warn] rate limited, retry {attempt + 1}/{max_retries} after {sleep_s:.1f}s: {exc}")
            time.sleep(sleep_s)


# Map language name → tree-sitter package name (reused from symbolic_annot registry)
_TS_PACKAGE: dict[str, str] = {
    "Python": "tree_sitter_python",
    "Rust": "tree_sitter_rust",
    "Go": "tree_sitter_go",
    "C": "tree_sitter_c",
    "CPP": "tree_sitter_cpp",
    "C++": "tree_sitter_cpp",  # alternate spelling
    "Java": "tree_sitter_java",
    "JavaScript": "tree_sitter_javascript",
    "TypeScript": "tree_sitter_typescript",
    "C#": "tree_sitter_c_sharp",
    "CSharp": "tree_sitter_c_sharp",  # McEval uses "CSharp"
    "Ruby": "tree_sitter_ruby",
    "Kotlin": "tree_sitter_kotlin",
    "Swift": "tree_sitter_swift",
    "Scala": "tree_sitter_scala",
    "Haskell": "tree_sitter_haskell",
    "PHP": "tree_sitter_php",
    "Lua": "tree_sitter_lua",
    "Shell": "tree_sitter_bash",
    "R": "tree_sitter_r",
    "Elixir": "tree_sitter_elixir",
    "Erlang": "tree_sitter_erlang",
}

# Languages where a bare method/function needs a class wrapper to parse cleanly.
# Value: (prefix, suffix) strings to wrap around the code slice.
_TS_WRAP: dict[str, tuple[str, str]] = {
    "C#":   ("class __W__ {\n", "\n}"),
    "Java": ("class __W__ {\n", "\n}"),
}

_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">"}

# tree-sitter node types whose children form a (callee, args) call pattern
_CALL_NODE_TYPES = {
    "call_expression", "call", "function_call", "invocation_expression",
    "method_invocation", "method_call_expression",
    "object_creation_expression",  # C#: new Foo(...)
    "template_function",  # C++: foo<T>(...)
}

_SELECTOR_NODE_TYPES = {
    "selector_expression",       # Go: pkg.Func / obj.Method / val.Field
    "member_access_expression",  # C#: obj.Method
    "field_expression",          # Rust/C variants
    "qualified_name",            # Java/C# namespace or type member
    "scoped_identifier",         # C++/Rust A::B
}

# tree-sitter node types that represent a return statement
_RETURN_NODE_TYPES = {
    "return_statement", "return_expression",
}

# tree-sitter node types for typed parameters / local variable declarations
_TYPED_DECL_TYPES = {
    "local_variable_declaration", "variable_declaration",
    "typed_parameter", "parameter", "parameter_declaration",
    "formal_parameter",
    "let_declaration",                  # Rust
    "property_declaration",             # Kotlin
    "local_declaration_statement",      # C#: int x = 5;
    "field_declaration",                # C#: class field
    "foreach_statement",                # C#/Java: foreach (var x in ...)
    "declaration_expression",           # C#: out var x
    "declaration",                      # C/C++: int x = 5; at any scope
    "init_declarator",                  # C/C++: x = 5 inside a declaration
    "structured_binding_declaration",   # C++17: auto [a, b] = ...
    "var_declaration",                  # Go: var x int
    "var_spec",                         # Go: x int = value inside var block
    "short_var_declaration",            # Go: x := value (already in DECL_PARENTS too)
    "const_declaration",                # Go: const x = value
    "const_spec",                       # Go: x = value inside const block
}

_TYPE_TOKEN_NODE_TYPES = {
    "type_identifier", "primitive_type", "predefined_type", "generic_type",
    "qualified_type", "scoped_identifier", "qualified_name", "generic_name",
    "pointer_type", "slice_type", "array_type", "map_type", "channel_type",
    "function_type", "interface_type", "struct_type", "nullable_type",
}

_NAME_TOKEN_NODE_TYPES = {
    "identifier", "field_identifier", "simple_identifier", "variable_name",
    "type_identifier",
}

_SKIP_LEAF_TEXT = {"", ",", ";", "(", ")", "{", "}", "[", "]"}


@dataclass
class StructuralEdge:
    token_i_idx: int   # cue token index
    token_j_idx: int   # predicted token index
    reason: str        # 'bracket' | 'type' | 'return' | 'call' | 'loop'


class SyntacticCheckerTool:
    """
    Uses tree-sitter to extract directed structural edges:
        token[i]  causes / predicts  token[j]

    All edges have 100% structural confidence — no LLM involved.
    Feeds into NeuralAnnotator as grounding so the LLM can focus on
    semantic/dataflow edges only.
    """

    def get_edges(
            self,
            code: str,
            subwords: list[SubwordToken],
            language: str = "Python",
            char_offset: int = 0,
    ) -> list[StructuralEdge]:
        """
        Returns directed structural edges for `code`.
        Falls back gracefully to [] if tree-sitter is not installed.

        Parameters
        ----------
        code : str
            The text to parse with tree-sitter.  When called from
            AnnotatorAgent this is the *Incomplete Code block only*
            (a substring of the full instruction), NOT the full string.
        subwords : list[SubwordToken]
            All tokens of the **full** instruction string.  Their
            char_start / char_end are global offsets.
        char_offset : int
            The byte offset at which `code` starts inside the full
            instruction string.  tree-sitter reports byte positions
            relative to the start of `code`; adding char_offset converts
            them to global positions so they can be looked up in
            offset_to_idx (which is built from global subwords).
        """
        pkg_name = _TS_PACKAGE.get(language)
        if pkg_name is None:
            return []
        try:
            from tree_sitter import Language, Parser
            pkg = importlib.import_module(pkg_name)
            lang = Language(pkg.language())
            parser = Parser(lang)

            # C# / Java: a bare method outside a class is not a valid top-level
            # construct in tree-sitter's grammar — it parses as global_statement /
            # expression_statement rather than method_declaration, so typed-decl /
            # return / call edges are never found.  Always wrap unconditionally.
            wrap = _TS_WRAP.get(language)
            if wrap:
                prefix, suffix = wrap
                parse_code = prefix + code + suffix
                # char_offset_eff = char_offset - len(prefix) keeps global positions
                # correct: global = node.start_byte(in wrapped) + char_offset_eff
                #        = (len(prefix) + local) + (char_offset - len(prefix))
                #        = local + char_offset  ✓
                parse_char_offset = char_offset - len(prefix)
            else:
                parse_code = code
                parse_char_offset = char_offset

            tree = parser.parse(bytes(parse_code, "utf-8"))
            root = tree.root_node

        except (ImportError, Exception) as _e:
            import traceback
            traceback.print_exc()
            return []

        # offset_to_idx is keyed by GLOBAL char positions (from full subwords)
        offset_to_idx = self._build_offset_map(subwords)
        edges: list[StructuralEdge] = []
        self._walk(root, parse_code, offset_to_idx, subwords, edges, parse_char_offset)
        return edges

    # ── Build offset → token index map ───────────────────────────────────────

    @staticmethod
    def _build_offset_map(subwords: list[SubwordToken]) -> dict[int, int]:
        m: dict[int, int] = {}
        for i, sw in enumerate(subwords):
            for pos in range(sw.char_start, sw.char_end):  # char_end is already exclusive, no +1
                if pos not in m:
                    m[pos] = i
        return m

    # ── Resolve a tree-sitter byte span → (surface, token_idx) ───────────────

    @staticmethod
    def _resolve(
            start: int,
            end: int,
            code: str,
            offset_to_idx: dict[int, int],
            subwords: list[SubwordToken],
            char_offset: int = 0,
    ) -> int:
        """Returns token index, or -1 if not found.

        start / end are tree-sitter byte positions (relative to the parsed
        slice).  char_offset converts them to global positions before the
        lookup in offset_to_idx.
        """
        for offset in range(start + char_offset, end + char_offset):
            if offset in offset_to_idx:
                return offset_to_idx[offset]
        return -1

    # ── Main entry: bracket pass first (global), then structural walk ─────────

    def _walk(self, root, code, offset_to_idx, subwords, edges, char_offset=0):
        # Global passes (single traversal over the whole tree)
        self._bracket_edges(root, code, offset_to_idx, subwords, edges, char_offset)
        self._defuse_edges(root, code, offset_to_idx, subwords, edges, char_offset)
        # Per-node passes (visit each node once)
        self._walk_structural(root, code, offset_to_idx, subwords, edges, char_offset)

    def _walk_structural(self, node, code, offset_to_idx, subwords, edges, char_offset=0):
        if node.type in _SELECTOR_NODE_TYPES:
            self._selector_edges(node, code, offset_to_idx, subwords, edges, char_offset)

        if node.type in _CALL_NODE_TYPES:
            self._call_edges(node, code, offset_to_idx, subwords, edges, char_offset)

        if node.type in _RETURN_NODE_TYPES:
            self._return_edges(node, code, offset_to_idx, subwords, edges, char_offset)

        if node.type in ("function_declaration", "method_declaration", "method_definition"):
            self._method_receiver_edges(node, code, offset_to_idx, subwords, edges, char_offset)
            self._return_type_edges(node, code, offset_to_idx, subwords, edges, char_offset)

        if node.type == "short_var_declaration":
            self._short_var_type_assertion_edges(node, code, offset_to_idx, subwords, edges, char_offset)

        if node.type in _TYPED_DECL_TYPES:
            self._type_edges(node, code, offset_to_idx, subwords, edges, char_offset)

        for child in node.children:
            self._walk_structural(child, code, offset_to_idx, subwords, edges, char_offset)

    # ── Edge extractors ───────────────────────────────────────────────────────

    _CLOSE_TO_OPEN = {v: k for k, v in _BRACKET_PAIRS.items()}

    # C# generic angle brackets — handled separately from _BRACKET_PAIRS
    _ANGLE_OPEN = {"<"}
    _ANGLE_CLOSE = {">"}

    def _bracket_edges(self, root, code, offset_to_idx, subwords, edges, char_offset=0):
        """
        Single global pass over all nodes in document order.
        Handles both leaf bracket tokens (Python/C/Java style) and
        non-leaf block delimiters (C# { } which are internal tree nodes).
        Also handles C# generic < > inside type_argument_list.
        """
        stack: list[tuple[str, int]] = []  # (open_char, token_idx)

        def process_text(text, start_byte, end_byte):
            if text in _BRACKET_PAIRS:
                i = self._resolve(start_byte, end_byte, code, offset_to_idx, subwords, char_offset)
                if i != -1:
                    stack.append((text, i))
            elif text in self._CLOSE_TO_OPEN:
                target_open = self._CLOSE_TO_OPEN[text]
                for k in range(len(stack) - 1, -1, -1):
                    if stack[k][0] == target_open:
                        _, open_idx = stack.pop(k)
                        j = self._resolve(start_byte, end_byte, code, offset_to_idx, subwords, char_offset)
                        if j != -1 and open_idx != j:
                            edges.append(StructuralEdge(open_idx, j, "bracket"))
                        break

        def walk(node):
            text = code[node.start_byte:node.end_byte]

            if node.child_count == 0:
                # Leaf node — standard bracket matching
                process_text(text, node.start_byte, node.end_byte)

            else:
                # Non-leaf: C# { } block delimiters appear as single-char
                # nodes that still have children (the block body).
                # Match them by text when they are exactly one character.
                if len(text) == 1 and text in (_BRACKET_PAIRS.keys() | self._CLOSE_TO_OPEN.keys()):
                    process_text(text, node.start_byte, node.end_byte)

                # Generic/template angle brackets: < > inside type_argument_list (C#/Java)
                # or template_argument_list (C++)
                if node.type in ("type_argument_list", "template_argument_list"):
                    for child in node.children:
                        ch_text = code[child.start_byte:child.end_byte]
                        if ch_text == "<":
                            i = self._resolve(child.start_byte, child.end_byte,
                                              code, offset_to_idx, subwords, char_offset)
                            if i != -1:
                                stack.append(("<", i))
                        elif ch_text == ">":
                            for k in range(len(stack) - 1, -1, -1):
                                if stack[k][0] == "<":
                                    _, open_idx = stack.pop(k)
                                    j = self._resolve(child.start_byte, child.end_byte,
                                                      code, offset_to_idx, subwords, char_offset)
                                    if j != -1 and open_idx != j:
                                        edges.append(StructuralEdge(open_idx, j, "bracket"))
                                    break

                for child in node.children:
                    walk(child)

        walk(root)

    def _selector_name_node(self, node):
        """Return the selected/member name in an expression like pkg.Func or obj.Method."""
        field = (
            node.child_by_field_name("field")
            or node.child_by_field_name("name")
            or node.child_by_field_name("property")
        )
        if field is not None:
            return field
        for child in reversed(node.children):
            if child.type in ("field_identifier", "identifier", "simple_identifier", "property_identifier"):
                return child
        return None

    def _selector_edges(self, node, code, offset_to_idx, subwords, edges, char_offset=0):
        """Package/object/type token predicts the selected field or method name."""
        name_node = self._selector_name_node(node)
        base_node = (
            node.child_by_field_name("operand")
            or node.child_by_field_name("object")
            or node.child_by_field_name("receiver")
            or node.child_by_field_name("scope")
        )
        if base_node is None:
            # Go selector_expression children are typically: operand, '.', field_identifier.
            children = [c for c in node.children if code[c.start_byte:c.end_byte] != "."]
            if len(children) >= 2:
                base_node = children[0]
                name_node = name_node or children[-1]
        if base_node is None or name_node is None:
            return
        dst = self._resolve(name_node.start_byte, name_node.end_byte, code, offset_to_idx, subwords, char_offset)
        if dst == -1:
            return
        for src in self._leaf_token_indices(base_node, code, offset_to_idx, subwords, char_offset):
            if src != dst:
                edges.append(StructuralEdge(src, dst, "api"))

    def _callee_token_indices(self, callee_node, code, offset_to_idx, subwords, char_offset=0) -> list[int]:
        if callee_node is None:
            return []
        if callee_node.type in _SELECTOR_NODE_TYPES:
            name_node = self._selector_name_node(callee_node)
            if name_node is not None:
                idx = self._resolve(name_node.start_byte, name_node.end_byte, code, offset_to_idx, subwords, char_offset)
                return [idx] if idx != -1 else []
        idx = self._resolve(callee_node.start_byte, callee_node.end_byte, code, offset_to_idx, subwords, char_offset)
        return [idx] if idx != -1 else []

    def _call_edges(self, node, code, offset_to_idx, subwords, edges, char_offset=0):
        """Callee/method token predicts each argument token."""
        callee_node = (
                node.child_by_field_name("function")
                or node.child_by_field_name("name")
                or node.child_by_field_name("method")
                or (node.children[0] if node.children else None)
        )
        callee_indices = self._callee_token_indices(callee_node, code, offset_to_idx, subwords, char_offset)
        if not callee_indices:
            return

        args_node = (
                node.child_by_field_name("arguments")
                or node.child_by_field_name("argument_list")
        )
        if args_node is None:
            return
        for arg in args_node.children:
            if arg.type in (",", "(", ")", " "):
                continue
            actual = arg.child_by_field_name("expression") or arg
            for callee_idx in callee_indices:
                for arg_idx in self._leaf_token_indices(actual, code, offset_to_idx, subwords, char_offset):
                    if arg_idx != callee_idx:
                        edges.append(StructuralEdge(callee_idx, arg_idx, "call"))

    def _leaf_token_indices(self, node, code, offset_to_idx, subwords, char_offset=0) -> list[int]:
        out: list[int] = []

        def walk(n):
            if n.child_count == 0:
                text = code[n.start_byte:n.end_byte]
                if text in _SKIP_LEAF_TEXT:
                    return
                idx = self._resolve(n.start_byte, n.end_byte, code, offset_to_idx, subwords, char_offset)
                if idx != -1 and (not out or out[-1] != idx):
                    out.append(idx)
                return
            for c in n.children:
                walk(c)

        walk(node)
        return out

    def _first_token_index(self, node, code, offset_to_idx, subwords, char_offset=0) -> int:
        indices = self._leaf_token_indices(node, code, offset_to_idx, subwords, char_offset)
        return indices[0] if indices else -1

    def _return_expression_nodes(self, return_node) -> list:
        for child in return_node.children:
            if child.type == "expression_list":
                return [c for c in child.children if c.child_count > 0 or c.type not in {",", ";"}]
        return [
            c for c in return_node.children
            if c.type != "return" and c.type not in {",", ";"}
        ]

    def _return_edges(self, node, code, offset_to_idx, subwords, edges, char_offset=0):
        """The `return` keyword predicts every returned expression token."""
        return_kw = next(
            (c for c in node.children if code[c.start_byte:c.end_byte] == "return"),
            None,
        )
        if return_kw is None:
            return
        i = self._resolve(return_kw.start_byte, return_kw.end_byte, code, offset_to_idx, subwords, char_offset)
        if i == -1:
            return
        for expr in self._return_expression_nodes(node):
            for j in self._leaf_token_indices(expr, code, offset_to_idx, subwords, char_offset):
                if j != i:
                    edges.append(StructuralEdge(i, j, "return"))

    def _direct_function_name_node(self, node):
        for child in node.children:
            if child.type in ("identifier", "field_identifier", "simple_identifier"):
                return child
        return None

    def _type_leaf_indices(self, node, code, offset_to_idx, subwords, char_offset=0) -> list[int]:
        out: list[int] = []

        def walk(n):
            if n.child_count == 0:
                text = code[n.start_byte:n.end_byte]
                if text in _SKIP_LEAF_TEXT or text == ".":
                    return
                if n.type in _NAME_TOKEN_NODE_TYPES or text in {"*", "[]", "chan", "map"}:
                    idx = self._resolve(n.start_byte, n.end_byte, code, offset_to_idx, subwords, char_offset)
                    if idx != -1 and idx not in out:
                        out.append(idx)
                return
            if n.type in _TYPE_TOKEN_NODE_TYPES or n.type in {"parameter_declaration", "variadic_parameter_declaration"}:
                for c in n.children:
                    walk(c)
            else:
                for c in n.children:
                    if c.type in _TYPE_TOKEN_NODE_TYPES or c.type in {"parameter_declaration", "variadic_parameter_declaration"}:
                        walk(c)

        walk(node)
        return out

    def _parameter_type_indices(self, parameter_node, code, offset_to_idx, subwords, char_offset=0) -> list[int]:
        type_node = parameter_node.child_by_field_name("type")
        if type_node is not None:
            return self._type_leaf_indices(type_node, code, offset_to_idx, subwords, char_offset)

        children = list(parameter_node.children)
        if not children:
            return []
        # Go unnamed result: parameter_declaration -> pointer_type/type_identifier only.
        if children[0].type in _TYPE_TOKEN_NODE_TYPES or code[children[0].start_byte:children[0].end_byte] == "*":
            candidates = children
        else:
            # Named parameter/result: skip leading identifier names, keep the trailing type nodes.
            candidates = [c for c in children[1:] if c.type not in {",", ""}]
        out: list[int] = []
        for c in candidates:
            for idx in self._type_leaf_indices(c, code, offset_to_idx, subwords, char_offset):
                if idx not in out:
                    out.append(idx)
        return out

    def _parameter_name_count(self, parameter_node, code) -> int:
        children = list(parameter_node.children)
        count = 0
        for child in children:
            text = code[child.start_byte:child.end_byte]
            if child.type in ("identifier", "field_identifier", "simple_identifier", "variable_name"):
                count += 1
                continue
            if text == ",":
                continue
            break
        return max(1, count)

    def _result_type_groups(self, fn_node, code, offset_to_idx, subwords, char_offset=0) -> list[list[int]]:
        name_node = self._direct_function_name_node(fn_node)
        if name_node is None:
            return []
        block_node = next((c for c in fn_node.children if c.type in {"block", "body", "statement_block"}), None)
        after_name = False
        param_lists_seen = 0
        result_nodes = []
        for child in fn_node.children:
            if child is name_node:
                after_name = True
                continue
            if not after_name:
                continue
            if child is block_node:
                break
            if child.type == "parameter_list":
                param_lists_seen += 1
                # After the function/method name, the first parameter_list is args;
                # any later parameter_list is a result tuple. The Go receiver appears
                # before the method name, so it is not counted here.
                min_seen = 1
                if param_lists_seen <= min_seen:
                    continue
                result_nodes.append(child)
            elif param_lists_seen >= 1:
                if child.type in _TYPE_TOKEN_NODE_TYPES:
                    result_nodes.append(child)

        groups: list[list[int]] = []
        for result in result_nodes:
            if result.type == "parameter_list":
                params = [c for c in result.children if c.type in {"parameter_declaration", "variadic_parameter_declaration"}]
                if params:
                    for param in params:
                        group = self._parameter_type_indices(param, code, offset_to_idx, subwords, char_offset)
                        if group:
                            for _ in range(self._parameter_name_count(param, code)):
                                groups.append(group)
                else:
                    group = self._type_leaf_indices(result, code, offset_to_idx, subwords, char_offset)
                    if group:
                        groups.append(group)
            else:
                group = self._type_leaf_indices(result, code, offset_to_idx, subwords, char_offset)
                if group:
                    groups.append(group)
        return groups

    def _collect_return_statements(self, node) -> list:
        out = []
        nested_fn_types = {
            "function_declaration", "method_declaration", "method_definition",
            "func_literal", "function_literal", "lambda_expression",
        }

        def walk(n, *, is_root: bool = False):
            if not is_root and n.type in nested_fn_types:
                return
            if n.type in _RETURN_NODE_TYPES:
                out.append(n)
            for c in n.children:
                walk(c)

        walk(node, is_root=True)
        return out

    def _return_type_edges(self, node, code, offset_to_idx, subwords, edges, char_offset=0):
        """Function/method result types predict returned expressions by tuple position."""
        type_groups = self._result_type_groups(node, code, offset_to_idx, subwords, char_offset)
        if not type_groups:
            return
        for ret in self._collect_return_statements(node):
            exprs = self._return_expression_nodes(ret)
            for pos, expr in enumerate(exprs):
                if pos >= len(type_groups):
                    break
                dst_indices = self._leaf_token_indices(expr, code, offset_to_idx, subwords, char_offset)
                for src in type_groups[pos]:
                    for dst in dst_indices:
                        if src != dst:
                            edges.append(StructuralEdge(src, dst, "type"))

    def _method_receiver_edges(self, node, code, offset_to_idx, subwords, edges, char_offset=0):
        """Go/C# style receiver/class type predicts the method name."""
        method_name = self._direct_function_name_node(node)
        if method_name is None:
            return
        dst = self._resolve(method_name.start_byte, method_name.end_byte, code, offset_to_idx, subwords, char_offset)
        if dst == -1:
            return
        receiver = None
        if node.type == "method_declaration":
            receiver = next((c for c in node.children if c.type == "parameter_list"), None)
        if receiver is None:
            return
        for param in [c for c in receiver.children if c.type in {"parameter_declaration", "variadic_parameter_declaration"}]:
            for src in self._parameter_type_indices(param, code, offset_to_idx, subwords, char_offset):
                if src != dst:
                    edges.append(StructuralEdge(src, dst, "type"))

    def _short_var_type_assertion_edges(self, node, code, offset_to_idx, subwords, edges, char_offset=0):
        """Go: type assertion `out, _ := x.(T)` gives T -> out."""
        expr_lists = [c for c in node.children if c.type == "expression_list"]
        if len(expr_lists) < 2:
            return
        lhs_items = [c for c in expr_lists[0].children if c.type in _NAME_TOKEN_NODE_TYPES]
        rhs_items = [c for c in expr_lists[1].children if c.type not in {",", ""}]
        for lhs, rhs in zip(lhs_items, rhs_items):
            if rhs.type != "type_assertion_expression":
                continue
            dst = self._resolve(lhs.start_byte, lhs.end_byte, code, offset_to_idx, subwords, char_offset)
            if dst == -1:
                continue
            for src in self._type_leaf_indices(rhs, code, offset_to_idx, subwords, char_offset):
                if src != dst:
                    edges.append(StructuralEdge(src, dst, "type"))

    def _defuse_edges(self, root, code, offset_to_idx, subwords, edges, char_offset=0):
        """
        Global two-pass def-use analysis (replaces _loop_edges).
        Pass 1: collect all declaration sites → { surface → [token_idx, ...] }
        Pass 2: walk all leaf identifiers; emit decl_idx → use_idx for each match.
        Covers loop variables, let/var declarations, parameters, assignments.
        """
        DECL_PARENTS = _TYPED_DECL_TYPES | {
            "for_statement", "for_in_statement", "foreach_statement",
            "enhanced_for_statement", "for_expression",  # loop kinds
            "assignment", "assignment_expression", "short_var_declaration",
            "let_declaration", "variable_declarator", "local_variable_declaration",
            # C#-specific
            "local_declaration_statement", "declaration_expression",
            "using_statement",              # using (var x = ...)
            # C/C++-specific
            "declaration",                  # int x = 5;
            "init_declarator",              # x = 5 part
            "structured_binding_declaration",  # C++17: auto [a, b] = ...
            # Go-specific
            "var_declaration", "var_spec",
            "const_declaration", "const_spec",
            "range_clause",                 # Go: for k, v := range ...
        }
        IDENT_TYPES = {
            "identifier", "simple_identifier", "variable_name", "variable",
            "implicit_type",    # C#: var keyword used as type
            "field_identifier", # C: struct field access
        }

        # ── Pass 1: find declared names ───────────────────────────────────────
        decl_map: dict[str, list[int]] = {}  # surface → [token_idx]

        def collect_decl_name_nodes(node):
            if node.type == "short_var_declaration":
                lhs = next((c for c in node.children if c.type == "expression_list"), None)
                if lhs is not None:
                    return [c for c in lhs.children if c.type in IDENT_TYPES and code[c.start_byte:c.end_byte] != "_"]
            name_node = next((c for c in node.children if c.type in IDENT_TYPES), None)
            return [name_node] if name_node is not None else []

        def collect_decls(node):
            if node.type in DECL_PARENTS:
                for name_node in collect_decl_name_nodes(node):
                    surface = code[name_node.start_byte:name_node.end_byte]
                    idx = self._resolve(name_node.start_byte, name_node.end_byte,
                                        code, offset_to_idx, subwords, char_offset)
                    if idx != -1:
                        decl_map.setdefault(surface, []).append(idx)
            for child in node.children:
                collect_decls(child)

        collect_decls(root)

        # ── Pass 2: walk all leaves; emit decl → use ─────────────────────────
        def collect_uses(node):
            if node.child_count == 0 and node.type in IDENT_TYPES:
                surface = code[node.start_byte:node.end_byte]
                if surface in decl_map:
                    j = self._resolve(node.start_byte, node.end_byte,
                                      code, offset_to_idx, subwords, char_offset)
                    if j != -1:
                        for decl_idx in decl_map[surface]:
                            if decl_idx != j:
                                edges.append(StructuralEdge(decl_idx, j, "defuse"))
            for child in node.children:
                collect_uses(child)

        collect_uses(root)

    def _type_edges(self, node, code, offset_to_idx, subwords, edges, char_offset=0):
        """
        In a typed declaration, type annotation tokens predict the variable name.
        Direction: type → varname  (seeing the type, you expect a name to follow).
        Excludes method/function declarations — return type → function name is NOT
        a type annotation; it is part of the function signature.
        """
        # Skip function/method declarations entirely
        if node.type in (
                "method_declaration", "function_declaration",
                "constructor_declaration", "function_definition",
                "method_definition",  # JS/TS
                "func_literal",  # Go
                "function_item",  # Rust
        ):
            return

        # Resolve variable name node — strategy differs by language:
        # C/C++: name lives inside a `declarator` field (which may be nested)
        # Java/C#/Python: name is a direct `identifier` child
        declarator_node = node.child_by_field_name("declarator")
        if declarator_node is not None:
            # C/C++: unwrap nested declarators to find the innermost identifier
            # e.g. declaration → init_declarator → identifier
            cur = declarator_node
            while cur is not None:
                inner = cur.child_by_field_name("declarator")
                if inner is None:
                    break
                cur = inner
            name_node = next(
                (c for c in (cur.children if cur.child_count > 0 else [cur])
                 if c.type in ("identifier", "field_identifier", "simple_identifier")),
                cur if cur.type in ("identifier", "field_identifier") else None
            )
        else:
            # Java / C# / Python / Go style: identifier is a direct child
            name_node = next(
                (c for c in node.children
                 if c.type in ("identifier", "simple_identifier", "variable_name")),
                None
            )

        type_node = (
                node.child_by_field_name("type")
                or node.child_by_field_name("type_annotation")
                # C# fallback: first child that looks like a type
                or next((c for c in node.children
                         if c.type in ("predefined_type", "identifier", "generic_name",
                                       "nullable_type", "array_type", "qualified_name",
                                       "type_identifier", "scoped_identifier")), None)
        )
        if name_node is None or type_node is None:
            return

        j = self._resolve(name_node.start_byte, name_node.end_byte, code, offset_to_idx, subwords, char_offset)

        def collect_type_tokens(n):
            if n.type in ("identifier", "simple_identifier", "type_identifier",
                          "generic_type", "scoped_identifier", "predefined_type",
                          "generic_name", "qualified_name", "nullable_type"):
                yield n
            for c in n.children:
                yield from collect_type_tokens(c)

        for type_tok in collect_type_tokens(type_node):
            i = self._resolve(type_tok.start_byte, type_tok.end_byte, code, offset_to_idx, subwords, char_offset)
            if i != -1 and j != -1 and i != j:
                edges.append(StructuralEdge(i, j, "type"))


class AnnotatorAgent:
    """
    NeuralAnnotator as tool-calling agent。
      - get_structural_edges   : tree-sitter edges（SyntacticCheckerTool）
      - search_api_docs        : search API docs
      - emit_correlations      : submit
    """

    def __init__(self, language: str = "Python", max_rounds: int = 6):
        self.language = language
        self.max_rounds = max_rounds
        self.model = os.environ.get("ANNOTATE_MODEL", "gpt-4o-mini")
        self.client = get_thread_local_openai_client()
        self._syntactic_tool = SyntacticCheckerTool()


    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_structural_edges",
                "description": (
                    "Run tree-sitter on the code and return deterministic structural edges: "
                    "bracket pairs, def-use, call arguments, return values, type annotations. "
                    "Always call this first before semantic analysis."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "language": {"type": "string", "description": "Programming language, e.g. 'Python'"},
                    },
                    "required": ["language"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_api_docs",
                "description": (
                    "Look up documentation for a library function or type. "
                    "Query must be specific: include the library name, function name, "
                    "and what you want to know. "
                    "Good: 'torch.nn.Linear parameters and return type' "
                    "Bad: 'linear layer'"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "e.g. 'torch.nn.Linear arguments'"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "emit_correlations",
                "description": (
                    "Submit all token correlation edges you found — both structural edges missed by "
                    "the syntactic checker AND semantic/dataflow/api edges. "
                    "Do NOT re-emit edges already returned by get_structural_edges. "
                    "reason must be one of: bracket, defuse, call, return, type, dataflow, semantic, api. "
                    "  bracket  — matching delimiter pair the checker missed (e.g. generics, language-specific)\n"
                    "  defuse   — variable declaration → use site the checker missed\n"
                    "  call     — callee → argument edge the checker missed\n"
                    "  return   — return keyword → returned expression the checker missed\n"
                    "  type     — type annotation → variable name the checker missed\n"
                    "  dataflow — value flows from producer to consumer (not same-name binding)\n"
                    "  semantic — grammar-paired keywords: if/else, try/catch, throw+exception, switch/case, import/using→identifier, async/await\n"
                    "  api      — library usage pattern: open→close, malloc→free, acquire→release\n"
                    "i and j are token indices — use exact indices to distinguish duplicate surface tokens. "
                    "You MUST call this to finish."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "i": {"type": "integer"},
                                    "j": {"type": "integer"},
                                    "reason": {"type": "string", "enum": [
                                        "bracket", "defuse", "call", "return",
                                        "type", "dataflow", "semantic", "api"
                                    ]},
                                },
                                "required": ["i", "j", "reason"],
                            },
                            "description": "Edges NOT already in the structural set returned by get_structural_edges.",
                        }
                    },
                    "required": ["pairs"],
                },
            },
        },
    ]

    def _execute_tool(self, name: str, inputs: dict, code: str,
                      subwords: list[SubwordToken],
                      ts_code: str | None = None,
                      ts_char_offset: int = 0) -> str:
        if name == "get_structural_edges":
            # Parse only the target slice (Incomplete Code block) so tree-sitter
            # sees valid source; char_offset converts local byte positions back to
            # global subword indices.
            parse_text = ts_code if ts_code is not None else code
            edges = self._syntactic_tool.get_edges(
                parse_text, subwords, self.language, char_offset=ts_char_offset
            )
            return json.dumps([
                {"i": e.token_i_idx, "j": e.token_j_idx, "reason": e.reason}
                for e in edges
            ])

        elif name == "search_api_docs":
            # 接你原来的 DuckDuckGo/OpenAI web search
            from src.annotate.web_search import search_docs  # 你已有的实现
            return search_docs(inputs["query"], language=self.language)

        elif name == "emit_correlations":
            # 终止信号，直接返回，agent 循环会检测到
            return "OK"

        return "Unknown tool"

    # ── Agent main loop ──────────────────────────────────────────────────────────

    def annotate(
        self,
        code: str,
        subwords: list[SubwordToken],
        target_indices: set[int] | None = None,
        ts_code: str | None = None,
        ts_char_offset: int = 0,
    ) -> list[TokenCorrelation]:
        """
        Annotate token correlations for ``code``.

        Parameters
        ----------
        code : str
            Full instruction string passed to the LLM.  Indices are global.
        subwords : list[SubwordToken]
            All tokens of ``code``, produced by tokenize_code_for_annotation.
        target_indices : set[int] | None
            If provided, only emit pairs where both token indices are in this
            set (restricts output to the Incomplete Code block).
        ts_code : str | None
            The text to feed to tree-sitter — should be the Incomplete Code
            block only (valid source that parses cleanly).  If None, falls
            back to the full ``code`` string (original behaviour).
        ts_char_offset : int
            Byte offset of ``ts_code`` within ``code``.  tree-sitter reports
            local positions; adding this converts them to global indices so
            they match the global ``subwords`` list.
        """
        indexed = {i: sw.surface for i, sw in enumerate(subwords)}

        system = (
            f"You are a {self.language} code analysis expert specializing in token-level dependency analysis.\n"
            "Your task: identify DIRECTED PREDICTIVE token dependencies in the given code.\n"
            "A pair [i, j] means: token[i] CAUSES or PREDICTS token[j] — "
            "i.e. a programmer who has seen token[i] would EXPECT token[j] to appear.\n"
            "Directionality: i is the CUE, j is the CONSEQUENCE. i < j preferred, but i > j is allowed.\n\n"

            "═══ REASON TAXONOMY ═══\n\n"

            "  bracket   — Matching bracket/delimiter pair the syntactic checker MISSED.\n"
            "              '(' predicts ')'; '[' predicts ']'; '{' predicts '}'.\n"
            "              Generic angle brackets <T> in C++/C#/Java/Go.\n\n"

            "  defuse    — Variable declaration → use site the checker MISSED.\n"
            "              Parameters used deep in function body; loop vars used in body;\n"
            "              variables from outer scopes; destructured variables.\n\n"

            "  call      — Callee → argument edge the checker MISSED.\n"
            "              Chained calls obj.foo().bar(); lambda/fp calls; constructor new Foo(...).\n\n"

            "  return    — return keyword → returned expression the checker MISSED.\n"
            "              Implicit returns (Rust/Ruby/Kotlin); ternary returns; yield.\n\n"

            "  type      — Type annotation → variable name the checker MISSED.\n"
            "              Cast expressions; type assertions; generic type constraints.\n\n"

            "  dataflow  — *** PRIMARY TASK — emit MANY of these ***\n"
            "              Data flows from a value-producing token to a value-consuming token.\n"
            "              You MUST find ALL of these — the syntactic checker emits ZERO dataflow edges.\n"
            "              Concrete patterns to look for (emit ALL that apply):\n"
            "                · Assignment: RHS expression tokens → LHS variable (the variable is now 'loaded' with that value)\n"
            "                · Variable read: assignment LHS → every subsequent read of that variable\n"
            "                · Parameter → every use of that parameter inside the function body\n"
            "                · Function call result → variable it is stored in\n"
            "                · Loop init variable → loop condition → loop body uses\n"
            "                · Conditional expression → branch body tokens that depend on it\n"
            "                · Accumulator/counter update → next iteration read\n"
            "                · Index variable → array/list access that uses it\n"
            "              Example for `double distance = Math.Abs(numbers[i] - numbers[j])`:\n"
            "                Abs → distance (call result stored in distance)\n"
            "                numbers → distance (input flows into computation)\n"
            "                i → distance (index determines which element)\n"
            "                j → distance (index determines which element)\n"
            "                distance → threshold comparison (distance is consumed)\n\n"

            "  semantic  — Grammar-paired control-flow keywords NOT captured by other types.\n"
            "              Emit for ALL of these patterns:\n"
            "                · 'for' predicts loop variable (i, j, k)\n"
            "                · 'for' predicts loop body tokens\n"
            "                · 'if' predicts condition tokens and body tokens\n"
            "                · 'if' predicts 'else' / 'elif' / 'else if'\n"
            "                · 'try' predicts 'catch' / 'except' / 'finally'\n"
            "                · 'throw' / 'raise' predicts exception class\n"
            "                · 'switch' predicts 'case' / 'default'\n"
            "                · 'while' predicts condition and body tokens\n"
            "                · 'async' predicts 'await'\n"
            "                · 'return' predicts returned value tokens (if checker missed)\n\n"

            "  api       — API/library usage pattern dependency.\n"
            "              The syntactic checker emits ZERO api edges — you must find all of them.\n"
            "              Patterns:\n"
            "                · Library type token predicts methods called on it (e.g. List → Count, Add)\n"
            "                · open → read/close; malloc → free; lock.acquire → lock.release\n"
            "                · Math.Abs → the numeric result it produces\n"
            "                · Any stdlib/framework call predicts how its return value is used\n\n"

            "═══ WORKFLOW ═══\n"
            "  1. Call get_structural_edges — returns bracket/defuse/call/return/type edges.\n"
            "     These are seeded automatically; do NOT re-emit them.\n"
            "  2. Systematically scan for dataflow edges:\n"
            "     - List every variable assignment and parameter in the code.\n"
            "     - For each, emit all dataflow edges to downstream consumers.\n"
            "     - Aim for at least 1 dataflow edge per variable/parameter.\n"
            "  3. Systematically scan for semantic edges:\n"
            "     - List every control-flow keyword (for/if/while/try/switch).\n"
            "     - Emit semantic edges to their paired tokens.\n"
            "  4. Scan for api edges:\n"
            "     - List every library/stdlib call.\n"
            "     - Emit api edges for usage patterns.\n"
            "  5. Call emit_correlations with ALL edges found in steps 2-4.\n"
            "     You MUST emit at least one dataflow edge — if you find none, look harder.\n\n"

            "═══ RULES ═══\n"
            "  - Duplicate surface tokens are DISTINCT — always use exact integer indices.\n"
            "  - Tokens inside comments are already filtered — ignore them.\n"
            "  - Be exhaustive. Each token[j] may have multiple predicting token[i]s.\n"
            "  - Do NOT re-emit edges already in the structural set.\n"
            "  - Quantity matters: more correct edges = better annotation.\n"
        )

        user = (
            f"Code:\n```{self.language}\n{ts_code if ts_code is not None else code}\n```\n\n"
            "Step 1: Call get_structural_edges to seed the structural edges.\n"
            "Then systematically find ALL dataflow, semantic, and api edges."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        final_pairs = []

        for _ in range(self.max_rounds):
            response = _rate_limited_chat_completion(
                self.client,
                model=self.model,
                tools=self.TOOLS,
                messages=messages,
            )
            msg = response.choices[0].message
            messages.append(msg)

            finish_reason = response.choices[0].finish_reason

            if finish_reason == "stop":
                messages.append({"role": "user",
                                 "content": "You must call emit_correlations to submit your results. "
                                            "Do not end without calling it."})
                continue

            if not msg.tool_calls:
                continue

            tool_results = []
            done = False
            for tc in msg.tool_calls:
                name = tc.function.name
                inputs = json.loads(tc.function.arguments)

                result = self._execute_tool(name, inputs, code, subwords,
                                            ts_code=ts_code, ts_char_offset=ts_char_offset)

                if name == "get_structural_edges":
                    structural_edges = json.loads(result)

                    # seed final_pairs with structural edges immediately
                    final_pairs = [
                        {"i": e["i"], "j": e["j"], "reason": e["reason"]}
                        for e in structural_edges
                    ]

                    by_reason = {}
                    for e in structural_edges:
                        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
                    summary = (
                            f"{len(structural_edges)} structural edges seeded: "
                            + ", ".join(f"{v} {k}" for k, v in by_reason.items())
                            + ". Do NOT re-emit these.\n\n"
                            "Now do the following IN ORDER:\n"
                            "1. DATAFLOW: List every variable/parameter declared in the code. "
                            "For each one, emit dataflow edges from its assigned value to all downstream reads. "
                            "Also emit dataflow from each parameter to every use inside the function body.\n"
                            "2. SEMANTIC: List every control-flow keyword (for/if/while/try/switch). "
                            "Emit semantic edges to their paired tokens (loop var, condition, body, else, catch).\n"
                            "3. API: List every library/stdlib call. "
                            "Emit api edges for usage patterns (type → method, call → result usage).\n"
                            "Then call emit_correlations with ALL edges from steps 1-3."
                    )
                    result = (
                        f"Indexed tokens:\n{json.dumps(indexed)}\n\n"
                        f"Structural edges already seeded:\n{json.dumps(structural_edges)}\n\n"
                        f"Instructions:\n{summary}"
                    )

                elif name == "emit_correlations":
                    # append semantic edges on top of already-seeded structural ones
                    final_pairs += inputs.get("pairs", [])
                    done = True

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            messages.extend(tool_results)
            if done:
                break
        else:
            import warnings
            warnings.warn(
                f"AnnotatorAgent exhausted max_rounds={self.max_rounds} without calling emit_correlations. "
                "Returning empty list.",
                RuntimeWarning,
            )

        _VALID_SUBTYPES = {"bracket", "defuse", "call", "return", "type", "dataflow", "semantic", "api"}

        return [
            TokenCorrelation(
                token_i=indexed.get(p["i"], ""),
                token_j=indexed.get(p["j"], ""),
                source="Neural",
                subtype=p.get("reason", "semantic") if p.get("reason") in _VALID_SUBTYPES else "semantic",
                token_i_idx=p["i"],
                token_j_idx=p["j"],
            )
            for p in final_pairs
            if p["i"] in indexed and p["j"] in indexed and p["i"] != p["j"]
            # If a target region is specified, restrict to pairs entirely within it.
            # This keeps global indices intact while only emitting correlations for
            # the Incomplete Code block (not the docstring / instruction wrapper).
            and (
                target_indices is None
                or (p["i"] in target_indices and p["j"] in target_indices)
            )
        ]