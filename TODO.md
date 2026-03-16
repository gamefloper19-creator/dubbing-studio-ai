# TODO

## High Priority

- [ ] Add unit tests for core pipeline modules
- [ ] Add integration tests for end-to-end dubbing workflow
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Add error handling and retry logic for API calls (Gemini, Edge TTS)
- [ ] Implement proper logging throughout the pipeline with log levels
- [ ] Add input validation for video file formats and sizes

## Features

- [ ] Add support for additional TTS engines
- [ ] Implement voice cloning / reference audio matching
- [ ] Add real-time progress WebSocket updates for batch processing
- [ ] Support multi-speaker detection and per-speaker voice assignment
- [ ] Add preview mode (process first 30 seconds only)
- [ ] Implement caching for translated segments to avoid redundant API calls
- [ ] Add support for custom glossaries / terminology for domain-specific translations
- [ ] Add drag-and-drop reordering of batch queue items in GUI

## Infrastructure

- [ ] Add Docker support with Dockerfile and docker-compose
- [ ] Add configuration file support (YAML/TOML) in addition to environment variables
- [ ] Set up pre-commit hooks (linting, formatting)
- [ ] Add type checking with mypy
- [ ] Add code formatting with black/ruff
- [ ] Set up automated dependency updates (Dependabot)
- [ ] Add health check endpoint for deployed instances

## Documentation

- [ ] Add API documentation for each module
- [ ] Add contributing guidelines (CONTRIBUTING.md)
- [ ] Add changelog (CHANGELOG.md)
- [ ] Document TTS engine comparison and language coverage matrix
- [ ] Add architecture diagrams
- [ ] Add example usage with sample videos

## Performance

- [ ] Profile and optimize memory usage for large video files
- [ ] Add streaming support for large file processing
- [ ] Optimize batch processing scheduling
- [ ] Add GPU memory management for concurrent TTS generation
- [ ] Benchmark different Whisper model sizes vs accuracy tradeoffs

## Quality

- [ ] Improve timing alignment accuracy (target <200ms deviation)
- [ ] Add audio quality metrics and validation
- [ ] Implement A/B comparison tool for dubbing quality evaluation
- [ ] Add subtitle timing validation against audio segments
