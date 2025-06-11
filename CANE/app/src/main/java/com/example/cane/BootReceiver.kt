package com.example.cane

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.os.Build
import androidx.annotation.RequiresApi

class BootReceiver : BroadcastReceiver() {
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) {
            val prefs = context.getSharedPreferences("tts_prefs", Context.MODE_PRIVATE)
            if (prefs.getBoolean("service_running", false)) {
                val svcIntent = Intent(context, MySpeechService::class.java)
                context.startForegroundService(svcIntent)
            }
        }
    }
}
