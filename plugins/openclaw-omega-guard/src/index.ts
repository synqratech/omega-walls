import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

import { loadPluginConfig } from "./config.js";
import { registerOmegaHooks } from "./hooks.js";
import { OmegaApiClient } from "./omega-client.js";
import { registerWebFetchProvider } from "./webfetch.js";

export default definePluginEntry({
  id: "omega-openclaw-guard",
  name: "Omega OpenClaw Guard",
  description: "Trust-boundary hooks, tool preflight, and guarded web fetch using Omega Walls API.",
  register(api: any) {
    const config = loadPluginConfig(api?.pluginConfig ?? {});
    const client = new OmegaApiClient(config);

    registerOmegaHooks(api, config, client);
    registerWebFetchProvider(api, config, client);
  }
});
