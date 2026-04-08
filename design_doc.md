# Bookly AI Support Agent — Design Document

---

## Production Readiness & Known Limitations

### Customer Authentication
The current prototype asks customers for their email or phone number at the start of each conversation, but this is entirely trust-based — there is no verification that the person providing the contact info actually owns the orders associated with it. Any user could type any email and retrieve someone else's order details.

In production, this would be replaced with a proper authentication layer:
- Each support session would be tied to a verified `customer_id` obtained at login (OAuth, magic link, session token, etc.)
- The `lookup_order` and `submit_refund` tools would accept `customer_id` as a parameter and filter results server-side, ensuring a customer can only access their own orders
- The agent would never need to ask for identifying information — the session context would already carry it

This is a deliberate tradeoff in the prototype: collecting identity via conversation is sufficient to demonstrate multi-turn interaction and is easy to mock, but it is not a security control.
