import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: './index.html',
        certificate:'./certificate.html'
        // Add more HTML files as needed
      }
    }
  }
})
