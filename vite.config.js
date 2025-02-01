import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: './index.html',
        certificate:'./certificate.html',
        module:'./module.html',
        recommendation:'./recommendation.html',
        project:'projects.html'
        // Add more HTML files as needed
      }
    }
  }
})
