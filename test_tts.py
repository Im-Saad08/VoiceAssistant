import asyncio
import edge_tts

TEXT = "Hello. I can hear you, and now I can speak to you as well."

async def main():
    communicate = edge_tts.Communicate(
        TEXT,
        "en-US-GuyNeural"
    )

    await communicate.save("test_voice.mp3")

asyncio.run(main())