def get_remote_address(request):
    return request.client.host if request.client else "unknown"
