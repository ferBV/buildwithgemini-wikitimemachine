import asyncio, json, os, uuid
import google.auth, google.auth.transport.requests, httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, Part, Role, TextPart, TransportProtocol

creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
auth_req = google.auth.transport.requests.Request()
creds.refresh(auth_req)

RESOURCE = 'projects/804344868103/locations/us-east1/reasoningEngines/3757125790110580736'
LOCATION = 'us-east1'
AGENT_DIRECTORY = 'app'
A2A_BASE = f'https://{LOCATION}-aiplatform.googleapis.com/reasoningEngines/v1/{RESOURCE}/api/a2a/{AGENT_DIRECTORY}'
A2A_CARD_URL = f'{A2A_BASE}/.well-known/agent-card.json'

async def test():
    headers = {'Authorization': f'Bearer {creds.token}'}
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        resp = await client.get(A2A_CARD_URL)
        card = AgentCard(**resp.json())
        card.url = A2A_BASE
        factory = ClientFactory(ClientConfig(supported_transports=[TransportProtocol.jsonrpc, TransportProtocol.http_json], httpx_client=client))
        a2a_client = factory.create(card)
        msg = Message(message_id=str(uuid.uuid4()), role=Role.user, parts=[Part(root=TextPart(text='Hola'))])
        async for ev in a2a_client.send_message(msg):
            print('EVENT:', repr(ev))

asyncio.run(test())
