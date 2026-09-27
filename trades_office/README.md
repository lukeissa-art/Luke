# TradeDesk AI: AI front office for trades shops

An AI receptionist for small plumbing, HVAC and electrical shops. It answers every call in the shop's
name, handles emergencies with fixed rules, books jobs into the shop's calendar, and sends the follow-up
texts that small shops never get around to. Flat monthly price. The business plan is in
[`docs/business-plan.md`](docs/business-plan.md).

"TradeDesk AI" is a placeholder name. Change `COMPANY_NAME` in `.env`.

## What's built (the MVP from the plan)

| Plan promise | Where it lives |
|---|---|
| **AI receptionist**: answers 24/7 in the shop's name, triages emergencies, collects name/number/address/problem, texts the owner a summary | `app/receptionist.py`, `app/emergency.py`, `app/telephony.py`, `app/calls.py` |
| **Booking**: offers open arrival windows from the shop's hours (and Google Calendar), confirms by text | `app/scheduling.py` |
| **Follow-up**: texts unbooked callers, nudges open quotes, asks for Google reviews after jobs, maintenance reminders | `app/followups.py`, `scripts/worker.py` |
| **Weekly report**: calls answered, after-hours calls, jobs booked, estimated revenue captured | `app/reports.py` |
| **Plans**: Starter $249 / Pro $449 / Plus $799, call caps, $500 setup fee, 14-day trial | `app/plans.py`, `app/billing.py` (Stripe) |
| **Founder dashboard**: onboard shops on the setup call, watch calls and transcripts, mark jobs done, MRR vs the 60-shop goal | `app/admin.py`, `app/templates/` |
| **Live demo for sales visits**: talk to a shop's assistant in the browser (typing or mic) | `/admin/shops/{id}/demo` |

### How a call works

1. The shop forwards calls to its Twilio number: all calls for 24/7 answering, or no-answer and after-hours forwarding on Starter.
2. Twilio hits `/voice/incoming`, the assistant greets the caller (with a recording and AI disclosure), and Twilio transcribes each reply.
3. Before the AI sees what the caller said, **fixed emergency rules** run. For gas, carbon monoxide or sparks, the caller gets safety instructions and is transferred to the owner's cell right away. The AI never makes that call. Flooding, no heat and similar get flagged, the owner gets an URGENT text, and the assistant offers a transfer.
4. Otherwise Claude runs the conversation with five tools: `check_availability`, `book_appointment`, `record_caller_details`, `transfer_to_owner` and `end_call`.
5. When the call ends, the owner gets a text summary. If the caller didn't book, a follow-up text goes out after 15 minutes (never during quiet hours) and is cancelled if they book in the meantime.
6. If the AI fails for any reason (outage, bad key), the caller is transferred to the owner. Nobody gets stuck on a broken line.

The owner can also text the assistant line:
- `DONE 12` marks appointment #12 complete and queues the review request
- `QUOTE 5125550123 1850 water heater` starts quote follow-ups
- `PAUSE` / `RESUME` turns the assistant off and on

## Run it locally (about 5 minutes)

```bash
cd trades_office
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # set ADMIN_PASSWORD and ANTHROPIC_API_KEY; leave Twilio blank for now
export DATABASE_PATH=trades_office.db
python -m scripts.seed_demo     # creates "Riverside Plumbing & Heating"
uvicorn app.main:app --reload
```

- Dashboard: http://localhost:8000/admin/ (log in with ADMIN_USERNAME / ADMIN_PASSWORD)
- Talk to the assistant in the terminal: `python -m scripts.simulate_call`
- Or in the browser: open the shop, then **Test call in browser**
- Tests: `python -m pytest -q` (no API keys needed; Claude is faked)

With Twilio left blank, texts are logged to the console and saved in the `sms_log` table instead of being sent.

## Go live

1. **Deploy** the `Dockerfile` (Railway, Render or Fly). Run two processes: `web` and `worker` (see `Procfile`). Put `DATABASE_PATH` on a persistent volume.
2. **Claude**: set `ANTHROPIC_API_KEY`. The default model is `claude-opus-5` at `low` effort, which keeps phone replies quick. If callers notice pauses or quality slips, change `ANTHROPIC_MODEL` / `ANTHROPIC_EFFORT`. Server-side refusal fallbacks are on (`ANTHROPIC_FALLBACKS=true`).
3. **Twilio**: buy one local number per shop. Set its Voice webhook to `POST {PUBLIC_BASE_URL}/voice/incoming`, its status callback to `{PUBLIC_BASE_URL}/voice/status`, and its Messaging webhook to `POST {PUBLIC_BASE_URL}/sms/incoming`. **Register for A2P 10DLC** before texting customers (see `docs/compliance.md`).
4. **Stripe**: create three monthly prices (249, 449, 799) and set their IDs. Point a webhook at `{PUBLIC_BASE_URL}/billing/webhook` for `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted` and `invoice.payment_failed`.
5. **Google Calendar** (optional, per shop): `pip install google-api-python-client google-auth`, create a service account, set `GOOGLE_SERVICE_ACCOUNT_FILE`, and have each shop share their calendar with the service account's email. Then set the shop's calendar to `google` in the dashboard.

## Company docs

- [`docs/business-plan.md`](docs/business-plan.md): the plan (problem, product, market, pricing, financials, risks)
- [`docs/launch-plan-90-days.md`](docs/launch-plan-90-days.md): week-by-week plan with go/no-go gates
- [`docs/sales-playbook.md`](docs/sales-playbook.md): shop visits, discovery questions, demo script, objections, trial close
- [`docs/onboarding-checklist.md`](docs/onboarding-checklist.md): the 2-hour setup call, step by step
- [`docs/operations.md`](docs/operations.md): weekly transcript review, support, metrics to watch
- [`docs/compliance.md`](docs/compliance.md): call recording consent, texting rules (TCPA, A2P 10DLC), customer terms

## Not built yet (months 4 to 12 in the plan)

- **Quote writer** (photos + voice notes → priced estimate from the shop's price book). The quote *follow-up* is built; the writer is next.
- **Jobber / Housecall Pro booking**. `app/scheduling.py` is the one place to add them, next to Google Calendar.
- **Real-time voice streaming**. Today's turn-based flow (Twilio speech recognition → Claude → text-to-speech) is simple and reliable but has a pause of about 1 to 3 seconds per reply. A streaming voice platform can replace `app/telephony.py` later without changing the receptionist logic.
- Multi-location shops (Plus) are billed but share one schedule per shop record for now.
