# Omega Walls: Mathematical Specification (v1)

This document is the **complete mathematical skeleton** for implementing the Ω (Omega) safety layer:
- a **wall-space** of “impossibilities” (risk clusters),
- an **intent-aware projection** from content to wall-pressures,
- an **accumulating memory** (scar-mass),
- a **cocktail/synergy** effect across walls,
- a deterministic **terminator** predicate `Off`,
- plus attribution primitives needed for product reaction logic.

It is written to be implementable directly.

---

## 0. Notation

- \(K\): number of walls (v1: \(K=4\)).
- Walls indexed by \(k \in \{1,\dots,K\}\).
- Content items (documents/chunks): \(x \in \mathcal{X}\).
- At step \(t\), the agent receives a **packet**:
  \[
  X_t = (x_{t,1}, \dots, x_{t,n_t}) \in \mathcal{X}^{n_t}.
  \]
- Wall pressure vectors:
  \[
  v(x) = (v^{(1)}(x), \dots, v^{(K)}(x)) \in \mathbb{R}_{\ge 0}^K.
  \]
- Toxicity (penetration) vectors:
  \[
  p_t \in [0,1]^K.
  \]
- Scar-mass (state) vectors:
  \[
  m_t \in \mathbb{R}_{\ge 0}^K.
  \]

---

## 1. Wall space (taxonomy)

Define the wall-space as a disjoint union:
\[
\Omega = \bigsqcup_{k=1}^{K} \Omega^{(k)}.
\]

For v1 we fix \(K=4\) with the following meanings:

1. **\(\Omega^{(1)}\): Override / Instruction-Takeover**  
   Attempts to change instruction priority (ignore previous rules, role injection, “system/developer mode”, etc.)

2. **\(\Omega^{(2)}\): Secret Exfiltration / Disclosure**  
   Attempts to reveal confidential information (tokens, passwords, API keys, system prompt, credentials, etc.)

3. **\(\Omega^{(3)}\): Tool / Action Abuse**  
   Attempts to force external actions (tool calls, commands, file/network writes), especially coupled with (1) or (2).

4. **\(\Omega^{(4)}\): Policy Evasion / Jailbreak**  
   Attempts to bypass or disable safety/policy constraints (“ignore safety”, “no restrictions”, “jailbreak”, etc.)

A single item \(x\) may pressure multiple walls simultaneously.

---

## 2. Projection: content \(\rightarrow\) wall pressure

### 2.1. Projector interface

A projector is a function
\[
\pi : \mathcal{X} \to \mathbb{R}_{\ge 0}^K,
\quad \pi(x)=v(x).
\]

For engineering, the projector must also emit **evidence** (matched spans/tokens, polarity, raw scores).  
Future learned projectors \(\pi_\theta\) must be drop-in compatible with the same output type.

### 2.2. Noise floor

Fix \(\varepsilon \ge 0\). Define componentwise flooring:
\[
\mathrm{floor}_\varepsilon(v) \;=\; \big(v^{(k)} \cdot \mathbf{1}\{v^{(k)} \ge \varepsilon\}\big)_{k=1}^{K}.
\]

All subsequent computations use
\[
\tilde v(x) := \mathrm{floor}_\varepsilon(\pi(x)).
\]

---

## 3. Packet composition (directive aggregation)

At step \(t\) compute per-item pressures:
\[
v_{t,j} := \tilde v(x_{t,j}) \in \mathbb{R}_{\ge 0}^K.
\]

**Non-compensating** composition:
\[
v_t := \bigoplus_{j=1}^{n_t} v_{t,j} \;\stackrel{\mathrm{v1}}{:=}\; \sum_{j=1}^{n_t} v_{t,j}.
\]

Interpretation: benign text never cancels malicious pressure.

---

## 4. Toxicity (penetration) mapping

Fix \(\alpha>0\). Define a monotone saturating map:
\[
\phi_\alpha(u) = 1 - e^{-\alpha u}, \quad u\ge 0.
\]

Base toxicity for the packet:
\[
p_t^{(0)} = \phi_\alpha(v_t) \in [0,1]^K,
\]
applied componentwise.

---

## 5. Cocktail / synergy across walls

Fix a nonnegative matrix
\[
S \in \mathbb{R}_{\ge 0}^{K\times K}, \quad S_{kk}=0,
\]
and a scalar \(\beta \ge 0\).

Define the synergy operator:
\[
(C_S(p))_k = \sum_{\ell \ne k} S_{k\ell}\,\min(p_k, p_\ell).
\]

Final toxicity (clipped to \([0,1]\)):
\[
p_t = \mathrm{clip}_{[0,1]}\Big(p_t^{(0)} + \beta\,C_S(p_t^{(0)})\Big).
\]

This encodes **superadditivity**: co-activated walls reinforce each other.

---

## 6. Scar deposit and memory dynamics

Define per-step deposit:
\[
e_t = v_t \odot p_t \in \mathbb{R}_{\ge 0}^K,
\]
where \(\odot\) is componentwise product.

Fix \(\lambda \in (0,1)\) (leak/decay). Update rule:
\[
m_{t+1} = \lambda m_t + e_t.
\]

Closed form (useful for analysis):
\[
m_t = \lambda^t m_0 + \sum_{i=0}^{t-1}\lambda^{t-1-i}e_i.
\]

Constant-pressure limit: if \(e_t \equiv e\), then
\[
m_t \to m^\star = \frac{1}{1-\lambda}e.
\]

---

## 7. Terminator predicate `Off`

Fix thresholds:
- \(\tau \in (0,1)\): spike threshold on toxicity,
- \(\Theta>0\): hard per-wall scar threshold,
- \(\Sigma>0\): total scar threshold,
- \(\theta>0\): per-wall threshold for multi-wall,
- \(N\in\{1,\dots,K\}\): required count of walls crossing \(\theta\).

Define
\[
\mathrm{Off}(m_{t+1}, p_t) = 1
\]
if any of the following holds:

1) **Spike**
\[
\max_k p_t^{(k)} \ge \tau.
\]

2) **Hard wall breach**
\[
\max_k m_{t+1}^{(k)} \ge \Theta.
\]

3) **Total mass breach**
\[
\sum_{k=1}^{K} m_{t+1}^{(k)} \ge \Sigma.
\]

4) **Multi-wall breach (distributed/cocktail)**
\[
\#\{k : m_{t+1}^{(k)} \ge \theta\} \ge N.
\]

Additionally, implementations SHOULD expose **reason flags** indicating which clause(s) fired.

---

## 8. Attribution (which item caused the Off)

For each item \(x_{t,j}\), compute a per-item deposit proxy using the shared packet toxicity \(p_t\):
\[
e_{t,j} = v_{t,j} \odot p_t,
\quad
c_{t,j} = \|e_{t,j}\|_1 = \sum_k e_{t,j}^{(k)}.
\]

Top contributor:
\[
j^\* = \arg\max_j c_{t,j}.
\]

For robust blocking, select all items with
\[
c_{t,j} \ge \gamma \cdot \max_i c_{t,i},
\]
with default \(\gamma = 0.7\).

This enables product policy to block 1–2 dominant docs, while handling “distributed within packet” attacks.

---

## 9. Baseline rule-based projector \(\pi_0\) (intent-aware v2)

### 9.1. Core principle: polarity gating

Each wall \(k\) has a local polarity \(\mathrm{pol}_k(x)\in\{-1,0,+1\}\).

- \(\mathrm{pol}_k=+1\): **directive** (“do X / reveal Y / ignore rules / bypass policy”)
- \(\mathrm{pol}_k=-1\): **prohibition/protection** (“do not reveal”, “never execute”, “avoid bypass”)
- \(\mathrm{pol}_k=0\): neutral discussion/education/examples without imperative intent

**v1 hard gate:**
\[
v^{(k)}(x) = \max(0,\mathrm{score}_k(x)) \cdot \mathbf{1}\{\mathrm{pol}_k(x)=+1\}.
\]

This is mandatory to prevent benign security content from accumulating scar mass.

### 9.2. Normalization

Compute two normalized views:
- \(t=\mathrm{norm}(x)\): lowercase, light homoglyph map (0→o,1→i,3→e,4→a,5→s,7→t,@→a,$→s), remove zero-width, collapse punctuation to spaces.
- \(t^\*=\mathrm{nospace}(t)\): remove spaces to detect spaced-letter obfuscations.

Let `tokens = t.split()`.

### 9.3. Windowed “action–object” intent

Define for token sets \(A,B\) and window \(w\):
\[
\mathrm{pair}_w(A,B)=1 \iff \exists a\in A, b\in B: \mathrm{dist}(a,b)\le w.
\]

Defaults: \(w=5\) for action–object pairing, \(w=6\) for negation/protection proximity.

### 9.4. Structural markers

Let `struct_count(x)` count matches of patterns like:
- `system:` / `developer:` (case-insensitive),
- code fences ``` ,
- `<system>` tags.

Structural markers provide only **small bonuses**.

---

### 9.5. Wall-specific scoring (v1 dictionaries + defaults)

Below, `count(S)` means the number of token hits from set `S` in `tokens` (or substring hits where indicated).

#### (1) Override / Instruction-Takeover

Dictionaries:
- anchors \(A_1\): {ignore, disregard, forget, override, follow, instructions, rules, system, developer, role, priority}
- phrases \(P_1\): canonical phrases (e.g., “ignore previous instructions”, …)
- fuzzy threshold \(f_{\min}=0.72\)

Rule: fuzzy phrase scoring is enabled only if at least one anchor is present.

Let:
- `anchor_hit = 1{count(A1)>0}`
- `best_phrase = max( fuzzy(t, p), fuzzy(t*, p*) )` over \(p\in P_1\)

Raw score:
\[
\mathrm{score}_1 =
0.8\cdot \mathrm{count}(A_1)\;+\;
2.2\cdot \mathbf{1}\{\text{anchor\_hit}\}\cdot \mathbf{1}\{\text{best\_phrase}\ge f_{\min}\}\cdot \text{best\_phrase}
\;+\;
0.5\cdot \mathrm{struct\_count}.
\]

Polarity \(\mathrm{pol}_1\) is +1 only if the text is imperative (override directive) and not negated (“do not ignore…”).

#### (2) Secret Exfiltration / Disclosure

Dictionaries:
- secret nouns \(N_2\): {api, key, token, password, secret, credentials, prompt}
- exfil verbs \(V_2\): {reveal, show, print, leak, exfiltrate, export, send, transmit, dump, share}
- protect verbs \(R\): {protect, secure, store, rotate, encrypt, redact, mask}
- negation markers \(G\): {"do not", "don't", "dont", "never", "avoid", "no", "without"}

Intent:
\[
I_{\text{exfil}} = \mathrm{pair}_5(V_2, N_2).
\]

Penalties:
- \(I_{\text{protect}} = \mathrm{pair}_5(R, N_2)\)
- \(I_{\text{neg}} = 1\) if any negation marker occurs within window 6 around a secret noun.

Raw score:
\[
\mathrm{score}_2 =
2.2\,I_{\text{exfil}} + 0.55\,\mathrm{count}(V_2) + 0.25\,\mathrm{count}(N_2)
- 1.3\,I_{\text{protect}} - 1.1\,I_{\text{neg}}.
\]

Polarity \(\mathrm{pol}_2\) is +1 if exfil intent holds and no protection/negation applies; otherwise -1 or 0.

#### (3) Tool / Action Abuse

Dictionaries:
- tool tokens \(T_3\): {curl, wget, bash, shell, execute, run, command, script, python, tool, function, write, file, call, call_tool}

Raw score:
\[
\mathrm{score}_3 =
0.55\,\mathrm{count}(T_3)
+ 0.6\,I_{\text{exfil}}
+ 0.4\,\mathbf{1}\{\mathrm{score}_1>0\}
+ 0.2\,\mathrm{struct\_count}.
\]

Polarity \(\mathrm{pol}_3\) is +1 if imperative (“run/execute/call”) and not negated (“never run…”).

#### (4) Policy Evasion / Jailbreak

Markers \(M_4\) as substrings in \(t\):
- “jailbreak”, “bypass”, “no restrictions”, “anything now”, “ignore safety”, “without safety”

Let \(I_{\text{evas}}=1\) if any marker matches.

Raw score:
\[
\mathrm{score}_4 = 2.0\,I_{\text{evas}} + 0.3\,\mathrm{struct\_count}.
\]

Polarity \(\mathrm{pol}_4\) is +1 if directive to bypass/ignore safety, and -1 if it prohibits bypass.

---

## 10. Default Ω parameters (engineering starting point)

These are starting defaults; calibrate on your traffic.

- \(K=4\)
- \(\varepsilon = 0.10\)
- \(\alpha = 1.0\) (toxicity saturation)
- \(\beta = 0.5\)
- \(\lambda = 0.85\)
- `Off` thresholds:
  - \(\tau = 0.9\)
  - \(\Theta = 0.8\)
  - \(\Sigma = 0.9\)
  - \(\theta = 0.4\)
  - \(N = 2\)

Synergy matrix \(S\) (suggested nonzero entries):
- reinforce tool/exfil and override:
  - \(S_{2,3}, S_{3,2} > 0\)
  - \(S_{1,2}, S_{2,1} > 0\)
  - \(S_{1,3}, S_{3,1} > 0\)
- policy evasion reinforces override:
  - \(S_{4,1}, S_{1,4} > 0\)

All other entries may be 0 in v1.

---

## 11. Minimal reaction policy (mathematics-to-product bridge)

While reaction orchestration is a separate product doc, the **mathematical outputs required** are:

- `reason flags` for Off clause(s),
- snapshots \(p_t\), \(m_{t+1}\), and \(v_t\),
- per-item contributions \(c_{t,j}\) for attribution,
- projector evidence per item.

These enable deterministic action selection, e.g.:
- block top docs (using \(\gamma\)-rule),
- freeze tools if wall 3 participates,
- escalate to human if wall 2 participates.

---

## 12. CI invariants (what must never regress)

1) **Hard negatives**: benign security docs must satisfy
\[
\sum_k v^{(k)}(x)=0
\]
under \(\pi_0\) (after polarity gating).

2) **Red-team positives**: generated attacks must satisfy:
- at least the targeted walls have \(v^{(k)}(x)>0\),
- obfuscations/paraphrases remain detectable.

3) **Ω stability**: benign-only streams should not trigger `Off` within a chosen horizon; mixed streams should trigger `Off` quickly.

---

## 13. Reference step algorithm (implementable pseudocode)

Given packet \(X_t\):

1) For each item \(j\):
   - \(v_{t,j} \leftarrow \mathrm{floor}_\varepsilon(\pi(x_{t,j}))\)

2) Aggregate:
   - \(v_t \leftarrow \sum_j v_{t,j}\)

3) Toxicity:
   - \(p^{(0)} \leftarrow \phi_\alpha(v_t)\)
   - \(p_t \leftarrow \mathrm{clip}(p^{(0)} + \beta C_S(p^{(0)}))\)

4) Deposit:
   - \(e_t \leftarrow v_t \odot p_t\)

5) Update:
   - \(m_{t+1} \leftarrow \lambda m_t + e_t\)

6) Off:
   - compute reasons via thresholds on \(p_t\), \(m_{t+1}\)

7) Attribution:
   - \(e_{t,j} \leftarrow v_{t,j} \odot p_t\)
   - \(c_{t,j} \leftarrow \|e_{t,j}\|_1\)
   - choose top docs using \(\gamma\) rule.

---

End of document.


