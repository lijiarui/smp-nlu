
from .app import APP
from .log import LOG

def web():
    host = 'localhost'
    port = 10343

    LOG.info('NLU web start at http://{}:{}'.format(host, port))
    APP.run(
        host=host,
        port=port,
        debug=True,
        use_reloader=False,
        threaded=True
    )

web()