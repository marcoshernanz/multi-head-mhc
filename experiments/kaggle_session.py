"""Look into (or stop) a running Kaggle kernel session: things the kaggle CLI 2.2 does not do.

Usage (needs kaggle >= 2.2, e.g. the scratch venv's python):
  python experiments/kaggle_session.py log <slug> [seconds]   live log (stdout of the kernel) while it runs, for up to <seconds>
  python experiments/kaggle_session.py files <slug>           output files of the last ended session, and its id (from their URLs)
  python experiments/kaggle_session.py cancel <session id>    stop the session (outputs written so far stay downloadable)

Found in exp03: `kaggle kernels logs` only works after a session ends, but the API streams the live log
(GetKernelSessionLogsStream). ListKernelSessionOutput lists output files whose URLs contain the session id (/kf/<id>/),
but only once a session has ended (exp03's had, while its status still said RUNNING); it is empty during a healthy run.
"""

import json
import re
import sys
import threading
import os

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiCancelKernelSessionRequest,
    ApiGetKernelSessionLogsStreamRequest,
    ApiListKernelSessionOutputRequest,
)

OWNER = "marcoshernanz"


def live_log(kaggle, slug: str, seconds: float) -> None:
    request = ApiGetKernelSessionLogsStreamRequest()
    request.user_name = OWNER
    request.kernel_slug = slug
    request.wait_for_logs_url_seconds = 30
    response = kaggle.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
    if "json" in response.headers.get("Content-Type", ""):  # the session has ended: the saved log comes back whole
        for entry in json.loads(response.text):
            print(entry["data"], end="")
        return
    threading.Timer(seconds, lambda: os._exit(0)).start()  # the stream stays open while the kernel runs
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: {"):
            print(json.loads(line[len("data: "):])["data"], end="", flush=True)
        elif line and "END_OF_LOG" in line:
            break
    os._exit(0)


def files(kaggle, slug: str) -> None:
    request = ApiListKernelSessionOutputRequest()
    request.user_name = OWNER
    request.kernel_slug = slug
    response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
    ids = {m.group(1) for f in response.files if (m := re.search(r"/kf/(\d+)/", f.url))}
    print(f"session id(s): {sorted(ids)}; {len(response.files)} files")
    for f in response.files:
        print(" ", f.file_name)


def cancel(kaggle, session_id: int) -> None:
    request = ApiCancelKernelSessionRequest()
    request.kernel_session_id = session_id
    response = kaggle.kernels.kernels_api_client.cancel_kernel_session(request)
    print("cancel requested", f"(error: {response.error_message})" if response.error_message else "")


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    command, target = sys.argv[1], sys.argv[2]
    with api.build_kaggle_client() as kaggle:
        if command == "log":
            live_log(kaggle, target, float(sys.argv[3]) if len(sys.argv) > 3 else 60)
        elif command == "files":
            files(kaggle, target)
        elif command == "cancel":
            cancel(kaggle, int(target))


if __name__ == "__main__":
    main()
