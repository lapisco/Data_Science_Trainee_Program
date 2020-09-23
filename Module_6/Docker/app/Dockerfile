FROM node:12-alpine
# OS dependencies
RUN apk update && apk add bash
# Code dependencies
WORKDIR /app
COPY . .
RUN yarn install --production
CMD ["node", "src/index.js"]
