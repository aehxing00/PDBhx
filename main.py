"""LigPlot-CN 启动入口。"""

import os
import sys
import traceback

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from ligplot_cn.gui import main  # noqa: E402

if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            with open(os.path.join(BASE, "LigPlotCN_error.log"), "w",
                      encoding="utf-8") as fh:
                fh.write(traceback.format_exc())
        except OSError:
            pass
        raise
