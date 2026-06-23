import { mapOmegaDecision } from "./dist/hooks.js";
const decision = mapOmegaDecision({ verdict: "block", reasons: ["smoke_attack"] });
if (!decision?.block) process.exit(1);
console.log(JSON.stringify({ status: "ok", decision }));
