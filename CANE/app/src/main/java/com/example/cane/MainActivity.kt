package com.example.cane

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.cane.ui.theme.CANETheme
import android.content.Intent
import android.content.SharedPreferences
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

class MainActivity : AppCompatActivity() {
    private lateinit var prefs: SharedPreferences

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        prefs = getSharedPreferences("tts_prefs", MODE_PRIVATE)

        val startBtn = findViewById<MaterialButton>(R.id.btn_start)
        val isRunning = prefs.getBoolean("service_running", false)
        startBtn.text = if (isRunning) "실행 중 (재등록 불필요)" else "서비스 시작"
        startBtn.isEnabled = !isRunning

        startBtn.setOnClickListener {
            // 초기화
            prefs.edit().putInt("last_read_index", 0)
                .putBoolean("service_running", true)
                .apply()
            // 포그라운드 서비스 시작
            val intent = Intent(this, MySpeechService::class.java)
            startForegroundService(intent)
            startBtn.text = "실행 중 (재등록 불필요)"
            startBtn.isEnabled = false
        }
    }
}