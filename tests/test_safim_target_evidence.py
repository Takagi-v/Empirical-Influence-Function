from src.annotate.target_evidence_annot import (
    annotate_completion_simple,
    annotate_prompt_simple,
    qwen_to_attention_edges,
)


def _edges_by_subtype(edges):
    out = {}
    for edge in edges:
        out.setdefault(edge.subtype, []).append((edge.token_i, edge.token_j))
    return out


def test_cpp_member_assignment_gets_target_centric_evidence_edges():
    full_code = """class a {};
class b {
private:
  int x;
public:
  a add(int x) {
     int y = 0;
     if (x == 0) { y = 2; }
     a z = y + 2;
     return z;
  }
};"""
    target = "     a z = y + 2;\n"
    target_start = full_code.index(target)
    target_end = target_start + len(target)

    tokens, edges = annotate_completion_simple(
        full_code=full_code,
        target_start=target_start,
        target_end=target_end,
        language="CPP",
        task_type="algorithmic_block",
    )

    by_subtype = _edges_by_subtype(edges)
    assert ("a", "a") in by_subtype["class"]
    assert ("add", "z") in by_subtype["function"]
    assert ("return", "z") in by_subtype["semantics"]
    assert ("y", "y") in by_subtype["declaration"]
    assert ("y", "z") in by_subtype["dataflow"]

    completion_indices = {
        i
        for i, token in enumerate(tokens)
        if token.char_start < target_end and token.char_end > target_start
    }
    assert all(edge.token_j_idx in completion_indices for edge in edges)


def test_api_call_gets_class_function_and_dataflow_evidence():
    full_code = """#include <fstream>
#include <string>

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    std::string content = load_text(path);
    return content;
}"""
    target = "load_text(path)"
    target_start = full_code.index(target)
    target_end = target_start + len(target)

    _, edges = annotate_completion_simple(
        full_code=full_code,
        target_start=target_start,
        target_end=target_end,
        language="CPP",
        task_type="api_function_call",
    )

    by_subtype = _edges_by_subtype(edges)
    assert ("read_file", "load_text") in by_subtype["function"]
    assert ("path", "path") in by_subtype["declaration"]
    assert ("content", "load_text") in by_subtype["dataflow"]


def test_semantics_do_not_cross_unrelated_python_functions():
    full_code = """def setup(self):
    if self.enabled:
        return self.value

def add_metric(name, data):
    labels = data.keys()
    values = [data[k] for k in labels]
    return values
"""
    target = "data.keys()"
    target_start = full_code.index(target)
    target_end = target_start + len(target)

    _, edges = annotate_completion_simple(
        full_code=full_code,
        target_start=target_start,
        target_end=target_end,
        language="Python",
        task_type="api_function_call",
    )

    noisy_semantics = [
        edge for edge in edges
        if edge.subtype == "semantics" and edge.token_i in {"if", "return"} and edge.token_i_idx < edge.token_j_idx
    ]
    assert noisy_semantics == []


def test_prompt_edges_are_exported_separately_from_completion_attention_edges():
    labels = [-100, -100, -100, 11, 12]
    qwen_annotations = [
        {"token_i_idx": 0, "token_j_idx": 2, "subtype": "declaration"},
        {"token_i_idx": 1, "token_j_idx": 3, "subtype": "dataflow"},
        {"token_i_idx": 3, "token_j_idx": 4, "subtype": "semantics"},
    ]

    attention_edges, prompt_edges = qwen_to_attention_edges(qwen_annotations, labels)

    assert attention_edges == [
        {"src": 1, "dst": 3, "subtype": "dataflow"},
        {"src": 3, "dst": 4, "subtype": "semantics"},
    ]
    assert prompt_edges == [
        {"src": 0, "dst": 2, "subtype": "declaration"},
    ]


def test_prompt_simple_edges_stay_inside_current_python_scope():
    full_code = """def setup(self):
    if self.enabled:
        return self.value

def add_metric(name, data):
    labels = data.keys()
    values = [data[k] for k in labels]
    return values
"""
    target = "data.keys()"
    target_start = full_code.index(target)
    target_end = target_start + len(target)

    tokens, prompt_edges = annotate_prompt_simple(
        full_code=full_code,
        target_start=target_start,
        target_end=target_end,
        language="Python",
        max_edges=64,
    )

    assert prompt_edges
    assert all(edge.token_i_idx < len(tokens) and edge.token_j_idx < len(tokens) for edge in prompt_edges)
    assert not any(edge.token_i == "if" and edge.token_i_idx < edge.token_j_idx for edge in prompt_edges)
    assert any(edge.subtype == "declaration" and edge.token_i == "data" and edge.token_j == "data" for edge in prompt_edges)


def test_python_module_owner_does_not_create_same_name_declaration_or_dataflow_fanout():
    full_code = """import os

def get_dirs(d):
    children = [os.path.join(d, child) for child in os.listdir(d)]
    dirs = filter(os.path.isdir, children)
    return list(dirs)

def gather_files(base_dir, file_mask):
    for dir_name, subdirs, files in os.walk(base_dir):
        full_mask = os.path.join(dir_name, file_mask)
        return full_mask
"""
    target = "os.path.join(dir_name, file_mask)"
    target_start = full_code.rindex(target)
    target_end = target_start + len(target)

    _, edges = annotate_completion_simple(
        full_code=full_code,
        target_start=target_start,
        target_end=target_end,
        language="Python",
        task_type="api_function_call",
    )

    os_edges = [
        edge for edge in edges
        if edge.token_j == "os" and edge.subtype in {"declaration", "dataflow"}
    ]
    assert os_edges == []
    assert any(edge.token_i == "os" and edge.token_j == "os" and edge.subtype == "class" for edge in edges)


def test_declaration_and_dataflow_do_not_overlap_on_same_token_pair():
    full_code = """def f(x):
    y = x + 1
    z = y + 2
    return z
"""
    target = "z = y + 2"
    target_start = full_code.index(target)
    target_end = target_start + len(target)

    _, edges = annotate_completion_simple(
        full_code=full_code,
        target_start=target_start,
        target_end=target_end,
        language="Python",
        task_type="algorithmic_block",
    )

    by_pair: dict[tuple[int, int], set[str]] = {}
    for edge in edges:
        by_pair.setdefault((edge.token_i_idx, edge.token_j_idx), set()).add(edge.subtype)
    assert all(not {"declaration", "dataflow"}.issubset(subtypes) for subtypes in by_pair.values())


def test_prompt_return_semantics_are_kept_before_dense_name_edges():
    full_code = """def gather_files(base_dir, file_mask):
    res_tuples = []
    for dir_name, subdirs, files in os.walk(base_dir):
        full_mask = os.path.join(dir_name, file_mask)
        mask_matches = glob(full_mask)
        res_tuples += [split_full_path(f, base_dir) for f in mask_matches]
        return pd.DataFrame(res_tuples, columns=['relative_dir', 'basename'])
"""
    target = "os.path.join(dir_name, file_mask)"
    target_start = full_code.index(target)
    target_end = target_start + len(target)

    _, prompt_edges = annotate_prompt_simple(
        full_code=full_code,
        target_start=target_start,
        target_end=target_end,
        language="Python",
        max_edges=64,
    )

    assert any(edge.token_i == "return" and edge.subtype == "semantics" for edge in prompt_edges)
