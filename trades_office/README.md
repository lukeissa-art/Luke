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
| **Public website**: home page with the pitch, pricing, a live "be the customer" demo, a free-trial signup form that lands in your dashboard (and texts you), privacy policy and terms | `app/site.py`, `app/templates/site/` |
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

- Website: http://localhost:8000/ (set `PUBLIC_DEMO_SHOP_ID=1` to switch on the live demo)
- Dashboard: http://localhost:8000/admin/ (log in with ADMIN_USERNAME / ADMIN_PASSWORD)
- Talk to the assistant in the terminal: `python -m scripts.simulate_call`
- Or in the browser: open the shop, then **Test call in browser**
- Tests: `python -m pytest -q` (no API keys needed; Claude is faked)

With Twilio left blank, texts are logged to the console and saved in the `sms_log` table instead of being sent.

## Go live

**Step-by-step hosting guide: [DEPLOY.md](DEPLOY.md)** (Render blueprint or Railway, about 15 minutes). The app runs as one service; follow-up jobs run inside it (`RUN_WORKER=true`).

1. **Deploy** the `Dockerfile` (Railway, Render or Fly). Run two processes: `web` and `worker` (see `Procfile`). Put `DATABASE_PATH` on a persistent volume.
2. **AI model**: set `GEMINI_API_KEY` to use Google Gemini (default model `gemini-3.8-flash`, change with `GEMINI_MODEL`), or `ANTHROPIC_API_KEY` to use Claude. If both are set, Gemini wins unless `LLM_PROVIDER=anthropic`. For Claude: The default model is `claude-sonnet-5` at `low` effort, which keeps phone replies quick and cheap. If replies feel shallow, raise `ANTHROPIC_EFFORT` to `medium`; to use a different model, set `ANTHROPIC_MODEL`. Server-side refusal fallbacks are off by default (`ANTHROPIC_FALLBACKS=false`); if Claude declines, the caller is told someone will call back.
3. **Twilio**: buy one local number per shop. Set its Voice webhook to `POST {PUBLIC_BASE_URL}/voice/incoming`, its status callback to `{PUBLIC_BASE_URL}/voice/status`, and its Messaging webhook to `POST {PUBLIC_BASE_URL}/sms/incoming`. **Register for A2P 10DLC** before texting customers (see `docs/compliance.md`).
4. **Stripe**: create three monthly prices (249, 449, 799) and set their IDs. Point a webhook at `{PUBLIC_BASE_URL}/billing/webhook` for `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted` and `invoice.payment_failed`.
5. **Google Calendar** (optional, per shop): create a Google service account, paste its key into `GOOGLE_SERVICE_ACCOUNT_JSON` (steps in DEPLOY.md), then use **Connect & test** on each shop's page after the owner shares their calendar. Busy time on the owner's calendar blocks slots; booked jobs are added to it.

## Website chat for your customers' sites

Each shop gets a chat button for its own website, using the same assistant as its phone line (the shop's prices, hours, services and calendar). On the shop's dashboard page, the **Website chat** box has:

- **A one-line snippet** to paste before `</body>` on the shop's site:
  `<script src="https://YOUR-APP/widget.js" data-shop="WIDGET_KEY" async></script>`
  (Wix: Settings → Custom code. Squarespace: Settings → Code injection → Footer. WordPress: a header & footer scripts plugin.) Optional `data-label="Book a visit"`, `data-color="#0f766e"`, `data-position="left"`.
- **A direct chat link** (`/chat/WIDGET_KEY`) for shops without a website: Google Business profile, Facebook, texts.

How it behaves:
- The chat opens in a small panel (an iframe, so the shop's site styles can't break it). It closes with × or Esc.
- It asks visitors for their phone number, books into open windows, and the owner gets a `[WEBSITE CHAT]` text summary. Chats that go quiet are wrapped up after 45 minutes, and visitors who left details but didn't book get the usual follow-up text.
- Emergencies (gas, CO, sparks) get safety instructions plus "call the shop now at …", and the owner gets an urgent text.
- Rate limited per visitor, counted toward the shop's monthly limit, and offline (with the shop's number) when the shop is paused or cancelled.

## Phone line for each shop

On a shop's dashboard page, under **Phone line**, type an area code and click **Get a phone number**. The app buys a local number on your Twilio account, points its call and text webhooks at this server, and saves it as the shop's assistant line (about $1.15/month plus usage on Twilio). There's also **Connect** for a number already on your Twilio account, and **Release number** to give one back.

The shop keeps its own number and forwards calls to the assistant line: all calls after hours, or just the ones they don't answer. The page lists the forwarding codes for Verizon, AT&T, T-Mobile and office phone systems. This needs `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` and a public https address. On Render, use a paid instance for real calls so the server never sleeps.

## The website

- `/` home page, `/privacy`, `/terms`, `/thanks`. The privacy policy includes the SMS wording Twilio's A2P 10DLC review looks for.
- **Trial signups** (`POST /trial`) are saved to the `leads` table, shown at the top of the dashboard, and texted to `FOUNDER_PHONE`. **Set up shop** on a signup pre-fills the onboarding form. The form works without JavaScript and has a hidden spam trap.
- **Live demo** (`POST /demo/say`) talks to the shop set in `PUBLIC_DEMO_SHOP_ID`. Each message is one Claude call, so it's limited to 20 messages per visitor per hour and 16 turns per call. Demo calls never text anyone and their bookings are released right away.
- Point your domain at the deployed app. Everything is served by the same process as the dashboard and phone webhooks.

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
