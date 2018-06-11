
import re


DOMAIN_SPLITOR = re.compile(r'\$\[(?P<domain_name>[^\]]+)\]')
SLOT_SPLITOR = re.compile(r'\$\{(?P<slot_name>[^\}]+)\}\((?P<slot_value>[^\)]+)\)')
SPLITOR = '___' # '_' * 3
