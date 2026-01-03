# Extension Backend

This is the backend for the extension 


## Usage
- rename a [example_config.yaml](./example_config.yaml) to config.yaml and modify it to suite your needs (for examples see [cli](../../cli/README.md)), if using docker all paths should be in a folder named resources
### To run using Docker
```bash
cd ..
cd ..
docker build -f extension/backend/Dockerfile . -t tarehimself/manga-translator-extension
```

### To run locally
```bash
uv sync
poe prod
```