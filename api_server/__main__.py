import uvicorn


if __name__ == '__main__':
    uvicorn.run('api_server.app:app', host='0.0.0.0', port=8000)
