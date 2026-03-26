FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

# Vite requires the --host flag to allow local network connections
CMD ["npm", "run", "dev", "--", "--host"]
