import asyncio
import configure

# Mock prompt to bypass interactive loop but execute internal logic
class MockPrompt:
    def ask(self): return "fake_key"
configure.questionary.password = lambda *args, **kwargs: MockPrompt()
configure.questionary.autocomplete = lambda *args, **kwargs: MockPrompt()

# Monkeypatch aiohttp to localhost
import aiohttp
original_get = aiohttp.ClientSession.get
def mock_get(self, url, headers=None):
    return original_get(self, "http://localhost:8080", headers=headers)
aiohttp.ClientSession.get = mock_get

# Ensure asyncio.run logic executes gracefully inside synchronous context
valid, models = asyncio.run(configure.check_api_key_and_get_models("fake"))
print("Parsed models synchronously isolated:", models)
