import sys
import os


def main():
    while True:
        sys.stdout.write("$ ")
        command = input()
        parts = command.split()
        if command == "exit":
            break
        elif parts[0] == "echo" :
            print(" ".join(parts[1:]))
        elif parts[0] == "type" :
            if parts[1] in ("echo", "type","exit"):
                print(f"{parts[1]} is a shell builtin")
            else :
                pathway = os.environ.get("PATH", "")
                found = False
                for dir_parth in pathway.split(os.pathsep):
                    full_path = os.path.join(dir_parth, parts[1])
                    if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                        print(f"{parts[1]} is {full_path}")
                        found = True
                        break
                if not found:
                    print(f"{parts[1]}: not found")
        else :
            path = os.environ.get("PATH", "")
            found = False
            for dir_parth in path.split(os.pathsep):
                full_path = os.path.join(dir_parth, parts[0])
                if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                    found = True
                    pid = os.fork()
                    if pid == 0:
                        os.execvp(full_path, parts)
                    else:
                        os.waitpid(pid,0)
                        break

            if not found:
                print(f"{parts[0]}: not found")

    pass

if __name__ == "__main__":
    main()
