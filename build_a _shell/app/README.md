# Mini Shell (Codecrafters)

A small Unix-like shell built in Python as a learning project (based on the Codecrafters "Build Your Own Shell" challenge).

The goal is to better understand how a shell works under the hood: parsing user input, searching executables in `PATH`, spawning processes, and managing I/O (redirections/pipelines coming next).

> ⚠️ Note: this project targets Unix/Linux (uses `fork/exec`). It is not intended to run on Windows.

---

## Current status

✅ Implemented
- Interactive prompt loop
- Basic built-ins: `echo`, `type`
- Execute external commands found in `PATH`
- Process creation + waiting (`fork/exec` + `waitpid`)
- Basic error handling (`command not found`)

🚧 In progress / next steps (Codecrafters roadmap)
- [ ] Quoting: single quotes, double quotes
- [ ] Escaping: backslash outside/inside quotes
- [ ] Redirections: `>`, `>>`, `2>`, `2>>`
- [ ] Pipelines: `cmd1 | cmd2` then multi-command pipelines
- [ ] History: `history`, navigation, execute entries
- [ ] History persistence: read/write history file
- [ ] Autocompletion: builtins + executables in `PATH`
