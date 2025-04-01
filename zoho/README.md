# Zoho Desk Integration

This folder contains the Deluge script to integrate Zoho Desk ticket replies with the RAG-powered AI response service.

## Script

**File:** `generate_ai_ticket_response.deluge`

This script:
- Retrieves an OAuth token.
- Sends the ticket description to the deployed RAG Compose API.
- Sends the AI-generated reply back to the ticket contact.

## Arguments

| Argument    | Description                                   |
|-----------  |-----------------------------------------------|
| ticketId    | Ticket ID in Zoho Desk.                       |
| description | Ticket description to generate a reply for.   |
| email       | Contact email to send the reply to.           |

## Notes

- This script requires valid OAuth credentials from Zoho.
- Configure the argument mapping for this function in Zoho Desk.
