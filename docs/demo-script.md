# Three-minute demo script

## 0:00 to 0:15 The problem

"Customer calls arrive as transcripts. A single prompt can pull out fields, but it
forgets the customer, cannot use company tools safely, and leaves no evidence
that it works."

## 0:15 to 0:45 The architecture

"Three agents share one typed state. The extractor reads the call, memory
retrieves prior interactions for the same customer, and action proposes scoped
tool calls. LangGraph pauses at a human approval gate. The executor writes
through a permission gateway, and every attempt is audited."

Show the graph on the Run tab. Point at extractor, memory, action, approval, and
executor.

## 0:45 to 2:15 The live run

1. Select "Acme Logistics: duplicate charge again".
2. Click Run.
3. Point at the graph as each node completes.
4. Show the extraction confidence and the memory hits from the August call.
5. Show the proposed refund and ticket.
6. Click "Approve and execute".
7. Show the audit timeline: proposed, approved, executed.

Say: "The memory agent found the prior resolution, and the action agent reused
it. The write only happened after a human approved the whole proposal."

## 2:15 to 2:45 Guardrails and evaluation

1. Run the low-confidence Atlas call. Show the escalation and the hidden approval.
2. Run the prompt-injection Northwind call. Show the denied refund or the
   policy escalation.
3. Open the Evaluation tab. Show system vs baseline, the router benchmark, and
   the failure cases.

Say: "The mesh improves memory recall and action accuracy over the stateless
baseline. The failures are documented instead of hidden."

## 2:45 to 3:00 Close

"The code, evaluation harness, and benchmark are in the repository. The demo runs
on free tiers with no external database. The graph, permissions, and router
policy are visible in the source."

Keep the browser at 100 percent zoom. Do not scroll faster than the viewer can
follow. Let the run finish before talking over the result.
