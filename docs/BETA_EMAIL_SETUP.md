# Beta email setup

Approved beta users get the day's top 10 trades at 7:30 AM on weekdays, with a link to the full list at [ericcosta.ca/finalyse](https://ericcosta.ca/finalyse).

People sign up on that page and confirm by email. You then approve each one in Buttondown.

## 1. Set up Buttondown

1. Create an account at [buttondown.com](https://buttondown.com) and set the newsletter name.
2. In the newsletter settings, add your name and a mailing address. Canada's anti-spam law (CASL) requires sender details in every message, and Buttondown adds them to the footer with the unsubscribe link.
3. Create two tags, with these exact names:
   - `beta-requested`, applied automatically to everyone who signs up
   - `beta-approved`, which you add to approve someone
4. Create an API key with subscriber and email sending access.

## 2. Add the key in two places

**The pipeline**, in `config/.env`:

```env
BUTTONDOWN_API_KEY=your_key
```

**The website**, as a secret on the `ericnc09-ericcosta-portfolio` Cloudflare Worker named `BUTTONDOWN_API_KEY`. Without it, the signup form tells visitors that signups aren't open yet.

## 3. Approve people

In Buttondown, open a confirmed subscriber who has `beta-requested` and add the `beta-approved` tag. They'll get the next morning's email.

The email only ever goes to people with `beta-approved`. If that tag doesn't exist, nothing is sent.

## Schedule

| Weekday time | What happens |
|---|---|
| 6:30 AM | Pipeline starts and publishes the day's trades by about 7:00 |
| 7:30 AM | `scripts/send_beta_email.py` sends the email |
| 5:30 PM | Pipeline runs again and updates the page |

The Mac must be awake at both morning times. The send refuses to go out if today's trades haven't been published, so a slow or failed run means no email rather than an old one.

Check what happened in `logs/beta_email.log`, looking for `beta_email.sent`.
