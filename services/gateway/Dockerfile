# Gateway: NestJS reverse proxy only (no business logic).
FROM node:20-slim AS build
WORKDIR /app
RUN corepack enable && corepack prepare yarn@1.22.22 --activate
COPY services/gateway/package.json services/gateway/tsconfig.json ./
COPY services/gateway/src ./src
RUN yarn install --frozen-lockfile || yarn install
RUN yarn build

FROM node:20-slim
WORKDIR /app
ENV NODE_ENV=production
RUN corepack enable && corepack prepare yarn@1.22.22 --activate
COPY services/gateway/package.json ./
RUN yarn install --production --frozen-lockfile || yarn install --production
COPY --from=build /app/dist ./dist
EXPOSE 8000
CMD ["node", "dist/main.js"]
