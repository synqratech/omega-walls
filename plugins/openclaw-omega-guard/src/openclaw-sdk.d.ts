declare module "openclaw/plugin-sdk/plugin-entry" {
  export function definePluginEntry<T extends object>(entry: T): T;
}
