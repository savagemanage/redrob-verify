import { Controller, Get } from "@nestjs/common";

@Controller()
export class MetaController {
  @Get("/health")
  health() {
    return { status: "ok", service: "gateway" };
  }

  @Get("/v1/meta")
  meta() {
    return {
      service: "gateway",
      version: "0.1.0",
      backend: "nestjs-proxy",
      model_sha256: "none",
      git_commit: process.env.GIT_COMMIT || "unknown",
      started_at: new Date().toISOString(),
    };
  }
}
