# gpus_lib.sh (runbook [z168], Fable board v20 rule 9; Runming 11-15: "运行脚本的多卡白名单里漏了……从机制上完善"): the ONE definition of a
# task's GPU count - its `#gpus <task> <n>` row in the node's queue file (1-8), else (Fable 11-15: a node-first arm is on the board before it
# has queue rows) its `<arm> #gpus <n>` line in tasks/board_arms.txt next to the queue file, else 1; capped by GPUS_MAX - used by
# run_tasks.sh (how many GPUs it gives a task) and by task.sh (it refuses any other count), so a multi-GPU task can no longer be missing
# from a second, hand-kept list (four times on 11-15: the 6K / 12K gates, PROBE-P5Q093T-4B, a 4B gate run on one GPU).
#   gpus_of_q <task> <queue file> [<cap>]   -> the count
#   gpus_check <task> <gpu list> <queue file> [<cap>]   -> prints why and returns 2 when the list's length is not the count
gpus_of_q() { local n; n=$(awk -v t="$1" '$1 == "#gpus" && $2 == t {print $3}' "$2" 2> /dev/null | tail -1)
  [ -n "$n" ] || n=$(awk -v t="$1" '$1 == t && $2 == "#gpus" {print $3}' "$(dirname "$2")/board_arms.txt" 2> /dev/null | tail -1)
  [[ $n =~ ^[1-8]$ ]] || n=1
  [[ ${3:-} =~ ^[1-8]$ ]] && [ $n -gt $3 ] && n=$3; echo $n; }
gpus_check() { local want got; want=$(gpus_of_q "$1" "$3" "${4:-}"); got=$(echo "$2" | tr , '\n' | grep -c .)
  [ "$got" = "$want" ] && return 0
  echo "$1 takes $want GPU(s) by its #gpus row in $(basename "$3") or board_arms.txt (no row = 1$([[ ${4:-} =~ ^[1-8]$ ]] && echo ", GPUS_MAX $4")), got $2"; return 2; }
