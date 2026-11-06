# WhatsApp alerts setup

Alerts go to Slack by default. Follow these steps to send them through the WhatsApp Cloud API instead.

This covers pipeline alerts and the two notify scripts. The interactive Slack `analyze` bot stays on Slack, because receiving WhatsApp messages needs a public webhook server.

## 1. Create the Meta app

1. In [Meta for Developers](https://developers.facebook.com/apps), create a Business app and add the WhatsApp product.
2. Under **WhatsApp > API Setup**, note the **Phone number ID** and add your number as a recipient.
3. In Business Settings, create a system user with `whatsapp_business_messaging` permission and generate a permanent token. The token on the API Setup page is temporary.

## 2. Submit the templates

WhatsApp only allows approved templates for messages the business starts. In WhatsApp Manager, create these with category **Utility** and language **English** (`en`).

The names and variable order must match exactly, or sends will fail.

### `conviction_alert`

```text
Smart money signal for {{1}}: {{2}}.
Actor: {{3}}
Conviction: {{4}}
Signal score: {{5}}
Fundamental score: {{6}}
Sector: {{7}}
Price: {{8}}
Research signal, not financial advice.
```

### `ensemble_alert`

```text
Ensemble alert for {{1}}: {{2}}.
Score: {{3}}
Confidence: {{4}}
Recommendation: {{5}}
Model components: {{6}}
Research signal, not financial advice.
```

### `top_pick`

```text
Morning top pick: {{1}}.
Composite score: {{2}}
Sector: {{3}}
Conviction: {{4}}
Ensemble: {{5}}
Probability of profit in 30 days: {{6}}
Smart-money buys: {{7}}
Latest signal: {{8}}
Runners-up: {{9}}
Research signal, not financial advice.
```

### `analysis_summary`

```text
Run complete: {{1}}.
Buy: {{2}}
Hold: {{3}}
Sell: {{4}}
Failed: {{5}}
Research signal, not financial advice.
```

## 3. Configure the repo

For local runs, add these to `config/.env`:

```env
ALERT_CHANNEL=whatsapp
WHATSAPP_TOKEN=your_permanent_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_TO=14165551234
```

`WHATSAPP_TO` is the recipient in international format, digits only.

For scheduled runs, add `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID` and `WHATSAPP_TO` as GitHub Actions **secrets**, then add `ALERT_CHANNEL=whatsapp` as a repository **variable**.

## 4. Send a test message

Once the templates are approved, run this with a small watchlist. It uses the `analysis_summary` template:

```bash
python scripts/analyze_and_notify.py RY
```

## Troubleshooting

- **Send failures:** the log event `whatsapp.send_failed` includes Meta's error message. Common causes are an unapproved template, a mismatched template name or variable count, or an expired token.
- **Accepted but never arrives:** `whatsapp.alert_accepted` only means Meta accepted the message. Delivery failures are reported through a webhook, which this repo doesn't run, so check the phone number and the template status in WhatsApp Manager.
- **Cost:** each template sent is billed per message by the recipient's country. See [Meta's pricing page](https://developers.facebook.com/docs/whatsapp/pricing).
