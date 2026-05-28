import base64, os

# stealth_eval.py content encoded in base64
b64 = (
"IiIiCkF0dGFjayBTdGVhbHRoIEV2YWx1YXRpb24gTW9kdWxlCgpFdmFsdWF0ZXMgdGhlIHN0"
"ZWFsdGggb2YgYWR2ZXJzYXJpYWwgYXR0YWNrcyBvbiBmYWN0LWNoZWNraW5nIHN5c3RlbXMK"
"YnkgY29tcHV0aW5nIHNlbWFudGljIHNpbWlsYXJpdHkgYmV0d2VlbiBjbGFpbXMgYW5kIGp1"
"c3RpZmljYXRpb25zIHVzaW5nClNlbnRlbmNlLUJFUlQgKGFsbC1tcG5ldC1iYXNlLXYyKS4K"
"IiIiCg=="
)

content = base64.b64decode(b64).decode('utf-8')
print(repr(content[:50]))
