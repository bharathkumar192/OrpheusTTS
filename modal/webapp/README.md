---
noteId: "c8836d103bae11f08c1507a08210f09e"
tags: []

---

# Maya Research - Orpheus Hindi TTS Web Application

A visually stunning web interface for Orpheus Hindi Text-to-Speech API, featuring a dark theme with green accents, parallax effects, and advanced audio synthesis options.

## Features

- Modern, responsive UI with stunning visual effects
- Hindi text-to-speech synthesis via Modal
- Waveform visualization for generated audio
- Sample Hindi sentences for quick testing
- Advanced synthesis parameters configuration
- System health status monitoring
- Audio file download capability

## Setup and Installation

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation Steps

1. Install dependencies:

```bash
cd modal/webapp
npm install
```

Or using yarn:

```bash
cd modal/webapp
yarn
```

2. Start the development server:

```bash
npm start
```

Or using yarn:

```bash
yarn start
```

The application will start on `http://localhost:3000` and proxy API requests to the Modal API server running on port 8000.

## Building for Production

To create a production build:

```bash
npm run build
```

Or using yarn:

```bash
yarn build
```

This will create an optimized build in the `build` directory that can be served by any static file server.

## API Integration

This webapp requires a running instance of the Orpheus TTS API provided by the Modal app. The API should expose the following endpoints:

- `/generate` - POST endpoint for text-to-speech generation
- `/health` - GET endpoint for checking API status

The proxy configuration in `package.json` assumes the API is running on `http://localhost:8000`. Modify the `proxy` field if your API is accessible at a different location.

## License

Copyright © 2023 Maya Research 
noteId: "d29ec1f0324411f0b9c70d17743ef7b7"
tags: []

---

 