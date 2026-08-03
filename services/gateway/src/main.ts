import "reflect-metadata";
import { NestFactory } from "@nestjs/core";
import { createProxyMiddleware } from "http-proxy-middleware";
import { AppModule } from "./app.module";

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { bodyParser: false });
  const expressApp = app.getHttpAdapter().getInstance();

  const upstream = {
    ocr: process.env.OCR_URL || "http://ocr:8001",
    face: process.env.FACE_URL || "http://face:8002",
    forgery: process.env.FORGERY_URL || "http://forgery:8003",
    identity: process.env.IDENTITY_URL || "http://identity:8004",
  };

  for (const [name, target] of Object.entries(upstream)) {
    expressApp.use(
      `/v1/${name}`,
      createProxyMiddleware({
        target,
        changeOrigin: true,
        // Keep path as-is (/v1/ocr/... → upstream /v1/ocr/...)
      }),
    );
  }

  const port = Number(process.env.GATEWAY_PORT || 8000);
  await app.listen(port, "0.0.0.0");
  console.log(`gateway listening on ${port}`);
}

bootstrap();
