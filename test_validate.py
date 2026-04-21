import urllib.request
from configure import check_api_key

print("Invalid key check:", check_api_key("gsk_invalid"))
# Can't easily test valid key unless Groq key is provided, but this confirms syntax and execution
