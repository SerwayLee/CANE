package com.example.cane

import android.app.*
import android.content.Intent
import android.content.SharedPreferences
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.media3.common.AudioAttributes
import okhttp3.*
import org.json.JSONObject
import java.io.IOException
import java.util.Locale
import java.util.UUID
import java.util.concurrent.LinkedBlockingQueue

class MySpeechService : Service(), TextToSpeech.OnInitListener {

    // ─── 멤버 ───────────────────────────────────────────────
    private lateinit var tts: TextToSpeech
    private lateinit var prefs: SharedPreferences
    private val client = OkHttpClient()
    private val handler = Handler(Looper.getMainLooper())
    private val queue = LinkedBlockingQueue<String>()

    private var isTtsReady = false
    private var isSpeaking = false

    // 3초마다 서버 폴링
    private val fetchRunnable = object : Runnable {
        override fun run() {
            fetchAndEnqueue()
            handler.postDelayed(this, 3_000)
        }
    }

    // ─── Service 생명주기 ──────────────────────────────────
    override fun onCreate() {
        super.onCreate()

        prefs = getSharedPreferences("tts_prefs", MODE_PRIVATE)

        tts = TextToSpeech(this, this)          // onInit() 대기
        startForeground(NOTI_ID, createNotification())

        handler.post(fetchRunnable)
    }

    override fun onDestroy() {
        handler.removeCallbacks(fetchRunnable)
        tts.shutdown()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // ─── TTS 초기화 ───────────────────────────────────────
    override fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) {
            Log.e("TTS", "TTS 초기화 실패 status=$status")
            stopSelf()
            return
        }
        tts.language = Locale.UK



        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) {
                isSpeaking = false
                handler.post { processQueue() }
            }
            override fun onError(utteranceId: String?) {
                isSpeaking = false
                handler.post { processQueue() }
            }
        })

        isTtsReady = true
        processQueue()                         // 대기열 처리 시작
    }

    // ─── 서버에서 문장 받아오기 (비동기) ─────────────────────
    private fun fetchAndEnqueue() {
        val req = Request.Builder()
            .url("https://port-0-delta-1cupyg2klvgk9x51.sel5.cloudtype.app/request")
            .build()

        client.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("TTS", "fetch 실패", e)
            }

            override fun onResponse(call: Call, res: Response) {
                if (!res.isSuccessful) return
                val arr = JSONObject(res.body!!.string()).getJSONArray("sentences")

                handler.post {
                    for (i in 0 until arr.length()) queue.offer(arr.getString(i))
                    Log.d("TTS", "받은 ${arr.length()}개, 대기열=${queue.size}")
                    processQueue()
                }
            }
        })
    }

    // ─── 큐에서 읽어서 speak() ──────────────────────────────
    private fun processQueue() {
        if (!isTtsReady) return
        if (isSpeaking || queue.isEmpty() || tts.isSpeaking) return

        val text = queue.poll() ?: return
        Log.d("TTS", "▶ speak: $text")

        val result = tts.speak(
            text,
            TextToSpeech.QUEUE_FLUSH,
            null,
            UUID.randomUUID().toString()
        )

        if (result == TextToSpeech.SUCCESS) {
            isSpeaking = true
        } else {
            Log.e("TTS", "speak 실패 code=$result")
            isSpeaking = false
            // 0.5초 후 재시도
            handler.postDelayed({ processQueue() }, 500)
        }
    }

    // ─── Foreground 알림 ──────────────────────────────────
    private fun createNotification(): Notification {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "TTS Background Service",
                NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java)
                .createNotificationChannel(channel)
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle("TTS 서비스 실행 중")
            .setContentText("서버 메시지를 자동으로 읽어줍니다")
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val CHANNEL_ID = "tts_service_channel"
        private const val NOTI_ID = 1
    }
}
