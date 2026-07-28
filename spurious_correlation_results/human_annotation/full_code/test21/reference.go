package main

func (cmd *Command) Uint(name string) uint {
	return getUint[uint](cmd, name)
}

func (cmd *Command) Uint8(name string) uint8 {
	return getUint[uint8](cmd, name)
}

func (cmd *Command) Uint16(name string) uint16 {
	return getUint[uint16](cmd, name)
}

func (cmd *Command) Uint32(name string) uint32 {
	return getUint[uint32](cmd, name)
}

func (cmd *Command) Uint64(name string) uint64 {
	return getUint[uint64](cmd, name)
}


func getUint[T uint | uint8 | uint16 | uint32 | uint64](cmd *Command, name string) T {
	if v, ok := cmd.Value(name).(T); ok { 		tracef("uint available for flag name %[1]q with value=%[2]v (cmd=%[3]q)", name, v, cmd.Name)

		return v 	}

	tracef("uint NOT available for flag name %[1]q (cmd=%[2]q)", name, cmd.Name)
	return 0
}