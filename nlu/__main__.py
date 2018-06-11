"""运行一个WEB服务器来提供NLU服务"""

from .app import APP
from .log import LOG

def web():
    host = 'localhost'
    port = 10343

    LOG.info('NLU web start at http://%s:%s', host, port)
    APP.run(
        host=host,
        port=port,
        debug=True,
        use_reloader=False,
        threaded=True)

web()
