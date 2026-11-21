package com.footsync.radio

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTimestamp
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.text.TextUtils
import android.util.Base64
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.media3.common.AudioAttributes
import androidx.media3.common.C
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.audio.AudioProcessor
import androidx.media3.common.audio.BaseAudioProcessor
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.datasource.HttpDataSource
import androidx.media3.exoplayer.DefaultLoadControl
import androidx.media3.exoplayer.DefaultRenderersFactory
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.Renderer
import androidx.media3.exoplayer.audio.AudioRendererEventListener
import androidx.media3.exoplayer.audio.AudioSink
import androidx.media3.exoplayer.audio.DefaultAudioSink
import androidx.media3.exoplayer.audio.ForwardingAudioSink
import androidx.media3.exoplayer.audio.MediaCodecAudioRenderer
import androidx.media3.exoplayer.hls.HlsMediaSource
import androidx.media3.exoplayer.mediacodec.MediaCodecSelector
import androidx.media3.exoplayer.metadata.MetadataOutput
import androidx.media3.exoplayer.source.MediaSource
import androidx.media3.exoplayer.source.ProgressiveMediaSource
import androidx.media3.exoplayer.text.TextOutput
import androidx.media3.exoplayer.upstream.DefaultLoadErrorHandlingPolicy
import androidx.media3.exoplayer.video.VideoRendererEventListener
import java.io.BufferedInputStream
import java.io.IOException
import java.io.InputStream
import java.net.ConnectException
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketTimeoutException
import java.net.URI
import java.net.URL
import java.net.UnknownHostException
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import javax.net.ssl.SSLException
import javax.net.ssl.SSLSocket
import javax.net.ssl.SSLSocketFactory
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

// =====================================================================
//  Delay engine: 16-bit PCM ring (125 s) + pre-delay amplitude envelope
// =====================================================================

class DelayAudioProcessor : BaseAudioProcessor() {

    companion object {
        const val MAX_DELAY_SEC = 120.0
        const val RING_SEC = 125
        const val ENV_HZ = 50
        const val ENV_CAP = (RING_SEC + 10) * ENV_HZ
        private const val FADE_SEC = 0.030
        private const val GAIN_RAMP_SEC = 0.020
    }

    class Snapshot(val env: FloatArray, val firstHop: Long, val epoch: Int)

    @Volatile private var targetDelaySec = 0.0
    @Volatile private var mutedFlag = false
    @Volatile private var clearReq = false
    @Volatile private var sampleRateV = 0
    @Volatile private var framesPublished = 0L
    @Volatile private var epochV = 0
    @Volatile private var noRingV = false

    // ---- playback-thread state ----
    private var sampleRate = 0
    private var channels = 0
    private var inputIsFloat = false
    private var ring: ShortArray? = null
    private var capFrames = 0
    private var writeIdx = 0
    private var curDelayFrames = 0
    private var fadeLeft = 0
    private var fadeLen = 1
    private var fadeFrom = 0
    private var fadeTo = 0
    private var gain = 1f

    // ---- envelope tap (guarded by envLock) ----
    private val envLock = Any()
    private val envRing = FloatArray(ENV_CAP)
    private var envHops = 0L
    private var framesTotal = 0L
    private var hopIdx = 0L
    private var hopStartFrame = 0L
    private var hopAcc = 0.0
    private var prevMono = 0.0
    private var framesPerHop = 1.0
    private var nextHopEnd = 1.0

    // ---------------- public API (thread-safe) ----------------

    fun setDelaySeconds(sec: Double) {
        targetDelaySec = sec.coerceIn(0.0, MAX_DELAY_SEC)
    }

    fun setMuted(m: Boolean) {
        mutedFlag = m
    }

    /** Next audio that arrives starts from an empty delay line (new stream). */
    fun requestClear() {
        clearReq = true
    }

    fun framesQueued(): Long = framesPublished

    fun currentSampleRate(): Int = sampleRateV

    fun epoch(): Int = epochV

    fun ringUnavailable(): Boolean = noRingV

    /** Seconds of pre-delay history available to the measurement. */
    fun historySeconds(): Double {
        synchronized(envLock) {
            return min(envHops, ENV_CAP.toLong()).toDouble() / ENV_HZ
        }
    }

    /** Most recent pre-delay envelope (ENV_HZ hops per second). */
    fun snapshot(maxSeconds: Double): Snapshot? {
        synchronized(envLock) {
            val total = envHops
            if (total <= 0L) return null
            val want = (maxSeconds * ENV_HZ).toLong()
            val n = min(want, min(total, ENV_CAP.toLong()))
            val first = total - n
            val cnt = n.toInt()
            val arr = FloatArray(cnt)
            for (k in 0 until cnt) {
                arr[k] = envRing[((first + k) % ENV_CAP).toInt()]
            }
            return Snapshot(arr, first, epochV)
        }
    }

    // ---------------- AudioProcessor ----------------

    override fun onConfigure(inputAudioFormat: AudioProcessor.AudioFormat): AudioProcessor.AudioFormat {
        val enc = inputAudioFormat.encoding
        if (enc != C.ENCODING_PCM_16BIT && enc != C.ENCODING_PCM_FLOAT) {
            throw AudioProcessor.UnhandledAudioFormatException(inputAudioFormat)
        }
        if (inputAudioFormat.channelCount < 1 || inputAudioFormat.sampleRate < 8000) {
            throw AudioProcessor.UnhandledAudioFormatException(inputAudioFormat)
        }
        // Output is always 16-bit: float input is converted when it enters the ring.
        return AudioProcessor.AudioFormat(
            inputAudioFormat.sampleRate,
            inputAudioFormat.channelCount,
            C.ENCODING_PCM_16BIT
        )
    }

    override fun onFlush() {
        val f = inputAudioFormat
        if (f.sampleRate > 0 && f.channelCount > 0) {
            val needAlloc = ring == null || f.sampleRate != sampleRate || f.channelCount != channels
            sampleRate = f.sampleRate
            channels = f.channelCount
            inputIsFloat = f.encoding == C.ENCODING_PCM_FLOAT
            sampleRateV = sampleRate
            if (needAlloc) allocate()
        }
        if (fadeLeft > 0) {
            curDelayFrames = fadeTo
            fadeLeft = 0
        }
    }

    override fun onReset() {
        ring = null
        capFrames = 0
        sampleRate = 0
        sampleRateV = 0
        channels = 0
        framesPublished = 0L
        synchronized(envLock) { envHops = 0L }
        epochV = epochV + 1
    }

    private fun targetFrames(): Int {
        val maxF = max(0, min((MAX_DELAY_SEC * sampleRate).toInt(), capFrames - 2))
        return Math.round(targetDelaySec * sampleRate).toInt().coerceIn(0, maxF)
    }

    private fun resetCounters() {
        writeIdx = 0
        framesTotal = 0L
        hopIdx = 0L
        hopStartFrame = 0L
        hopAcc = 0.0
        prevMono = 0.0
        framesPerHop = if (sampleRate > 0) sampleRate.toDouble() / ENV_HZ else 1.0
        nextHopEnd = framesPerHop
        synchronized(envLock) {
            java.util.Arrays.fill(envRing, 0f)
            envHops = 0L
        }
        framesPublished = 0L
        fadeLeft = 0
        gain = if (mutedFlag) 0f else 1f
        epochV = epochV + 1
    }

    private fun allocate() {
        val cap = RING_SEC * sampleRate
        var arr: ShortArray? = null
        try {
            arr = ShortArray(cap * channels)
        } catch (e: OutOfMemoryError) {
            arr = null
        }
        noRingV = arr == null
        ring = arr
        capFrames = if (arr == null) 0 else cap
        resetCounters()
        curDelayFrames = if (arr == null) 0 else targetFrames()
    }

    private fun putSample(out: ByteBuffer, v: Float) {
        var x = v
        if (x > 32767f) x = 32767f else if (x < -32768f) x = -32768f
        out.putShort(x.toInt().toShort())
    }

    override fun queueInput(inputBuffer: ByteBuffer) {
        val remaining = inputBuffer.remaining()
        if (remaining == 0) return
        val ch = channels
        val bpf = if (inputIsFloat) 4 * ch else 2 * ch
        if (ch <= 0 || bpf <= 0) {
            inputBuffer.position(inputBuffer.limit())
            return
        }
        val frames = remaining / bpf
        if (frames <= 0) {
            inputBuffer.position(inputBuffer.limit())
            return
        }
        val out = replaceOutputBuffer(frames * ch * 2)
        val r = ring
        var inPos = inputBuffer.position()

        if (r == null) {
            // Not enough memory for the delay line: plain pass-through.
            for (i in 0 until frames * ch) {
                if (inputIsFloat) {
                    putSample(out, inputBuffer.getFloat(inPos) * 32767f)
                    inPos += 4
                } else {
                    out.putShort(inputBuffer.getShort(inPos))
                    inPos += 2
                }
            }
            inputBuffer.position(inputBuffer.limit())
            out.flip()
            return
        }

        if (clearReq) {
            clearReq = false
            java.util.Arrays.fill(r, 0.toShort())
            resetCounters()
        }

        val cap = capFrames
        val sr = sampleRate
        val tgt = targetFrames()
        if (fadeLeft == 0 && tgt != curDelayFrames) {
            fadeFrom = curDelayFrames
            fadeTo = tgt
            fadeLen = max(32, (FADE_SEC * sr).toInt())
            fadeLeft = fadeLen
        }

        val muteGain = if (mutedFlag) 0f else 1f
        val gStep = 1f / max(16f, (GAIN_RAMP_SEC * sr).toFloat())

        for (i in 0 until frames) {
            // 1) store the live frame + accumulate the pre-delay envelope
            val wBase = writeIdx * ch
            var sum = 0
            for (c in 0 until ch) {
                val s: Int
                if (inputIsFloat) {
                    var v = inputBuffer.getFloat(inPos) * 32767f
                    inPos += 4
                    if (v > 32767f) v = 32767f else if (v < -32768f) v = -32768f
                    s = v.toInt()
                } else {
                    s = inputBuffer.getShort(inPos).toInt()
                    inPos += 2
                }
                r[wBase + c] = s.toShort()
                sum += s
            }
            val mono = sum.toDouble() / ch
            val d = mono - prevMono          // first difference = pre-emphasis
            prevMono = mono
            hopAcc += if (d < 0) -d else d
            framesTotal++
            if (framesTotal >= nextHopEnd) {
                val len = (framesTotal - hopStartFrame).toDouble()
                val e = (hopAcc / len / 32768.0).toFloat()
                synchronized(envLock) {
                    envRing[(hopIdx % ENV_CAP).toInt()] = e
                    envHops = hopIdx + 1
                }
                hopIdx++
                hopStartFrame = framesTotal
                hopAcc = 0.0
                nextHopEnd = (hopIdx + 1) * framesPerHop
            }

            // 2) read the delayed frame (short cross-fade while the delay changes)
            val dA = if (fadeLeft > 0) fadeFrom else curDelayFrames
            var ia = writeIdx - dA
            if (ia < 0) ia += cap
            if (fadeLeft > 0) {
                var ib = writeIdx - fadeTo
                if (ib < 0) ib += cap
                val a = 1f - fadeLeft.toFloat() / fadeLen
                for (c in 0 until ch) {
                    val va = r[ia * ch + c].toInt()
                    val vb = r[ib * ch + c].toInt()
                    putSample(out, (va + (vb - va) * a) * gain)
                }
                fadeLeft--
                if (fadeLeft == 0) curDelayFrames = fadeTo
            } else {
                for (c in 0 until ch) {
                    putSample(out, r[ia * ch + c].toInt() * gain)
                }
            }

            // 3) smooth mute / unmute
            if (gain != muteGain) {
                gain = if (gain < muteGain) min(muteGain, gain + gStep) else max(muteGain, gain - gStep)
            }
            writeIdx++
            if (writeIdx == cap) writeIdx = 0
        }

        inputBuffer.position(inputBuffer.limit())
        out.flip()
        framesPublished = framesTotal
    }
}

/** (PTS of the first sample of a buffer, processor frames consumed before it). */
class PtsPair(val ptsUs: Long, val framesBefore: Long)

/** Sink playhead sample: wall clock (System.nanoTime) + media position heard at that instant. */
class Playhead(val nanoTime: Long, val posUs: Long)

/**
 * Wraps the real DefaultAudioSink to learn the exact playout latency (processor -> speaker),
 * so a measured delay can be compared with the TV on one wall-clock axis.
 */
class SyncAudioSink(sink: AudioSink, private val proc: DelayAudioProcessor) : ForwardingAudioSink(sink) {

    @Volatile private var ptsPair: PtsPair? = null
    @Volatile private var playhead: Playhead? = null
    private var lastPtsUs = Long.MIN_VALUE

    override fun handleBuffer(buffer: ByteBuffer, presentationTimeUs: Long, encodedAccessUnitCount: Int): Boolean {
        if (presentationTimeUs != lastPtsUs) {
            lastPtsUs = presentationTimeUs
            ptsPair = PtsPair(presentationTimeUs, proc.framesQueued())
        }
        return super.handleBuffer(buffer, presentationTimeUs, encodedAccessUnitCount)
    }

    override fun getCurrentPositionUs(sourceEnded: Boolean): Long {
        val p = super.getCurrentPositionUs(sourceEnded)
        if (p != AudioSink.CURRENT_POSITION_NOT_SET) {
            playhead = Playhead(System.nanoTime(), p)
        }
        return p
    }

    override fun flush() {
        ptsPair = null
        playhead = null
        lastPtsUs = Long.MIN_VALUE
        super.flush()
    }

    override fun reset() {
        ptsPair = null
        playhead = null
        lastPtsUs = Long.MIN_VALUE
        super.reset()
    }

    /**
     * Wall-clock time (seconds, System.nanoTime base) at which processor frame 0 would be
     * heard on the speaker with a delay of 0 s. NaN when the sink has not reported recently.
     */
    fun r0Seconds(sampleRate: Int): Double {
        val pp = ptsPair ?: return Double.NaN
        val ph = playhead ?: return Double.NaN
        if (sampleRate <= 0) return Double.NaN
        val ageMs = (System.nanoTime() - ph.nanoTime) / 1_000_000L
        if (ageMs > 1000L) return Double.NaN
        val a = pp.ptsUs / 1e6 - pp.framesBefore.toDouble() / sampleRate
        return ph.nanoTime / 1e9 - ph.posUs / 1e6 + a
    }
}

/**
 * DefaultRenderersFactory whose only renderer is a MediaCodecAudioRenderer feeding our
 * DefaultAudioSink (+ DelayAudioProcessor). Overriding createRenderers keeps this independent of
 * buildAudioSink's parameter list, which changed between Media3 releases.
 */
class SyncRenderersFactory(private val ctx: Context, private val sink: AudioSink) : DefaultRenderersFactory(ctx) {
    override fun createRenderers(
        eventHandler: Handler,
        videoRendererEventListener: VideoRendererEventListener,
        audioRendererEventListener: AudioRendererEventListener,
        textRendererOutput: TextOutput,
        metadataRendererOutput: MetadataOutput
    ): Array<Renderer> {
        return arrayOf<Renderer>(
            MediaCodecAudioRenderer(
                ctx,
                MediaCodecSelector.DEFAULT,
                eventHandler,
                audioRendererEventListener,
                sink
            )
        )
    }
}

// =====================================================================
//  Local raw-socket relay: ICY / HTTP(S) upstream -> clean HTTP/1.1 for ExoPlayer
// =====================================================================

fun describeNetError(e: Throwable, hostPort: String): String {
    return when (e) {
        is StreamRelay.HttpStatusException -> {
            val why = when (e.code) {
                400 -> "طلب غير صالح"
                401, 403 -> "الوصول مرفوض (قد يتطلب البث اشتراكاً أو يحجب هذا التطبيق)"
                404 -> "الرابط غير موجود أو انتهت صلاحيته"
                429 -> "طلبات كثيرة جداً"
                in 500..599 -> "عطل في خادم البث"
                else -> "رد غير متوقع من الخادم"
            }
            "الخادم أعاد HTTP ${e.code} ${e.reason} — $why"
        }
        is UnknownHostException -> "تعذّر العثور على الخادم «$hostPort» — تحقق من الرابط ومن اتصال الإنترنت"
        is ConnectException -> "رفض الخادم الاتصال «$hostPort» — قد يكون البث متوقفاً أو المنفذ خاطئاً (${e.message ?: ""})"
        is SocketTimeoutException -> "انتهت مهلة الاتصال بالخادم «$hostPort» — لم يصل أي رد"
        is SSLException -> "فشل الاتصال الآمن (SSL) مع «$hostPort»: ${e.message ?: e.javaClass.simpleName}"
        else -> "${e.javaClass.simpleName}: ${e.message ?: "خطأ غير معروف"}"
    }
}

class StreamRelay(private val targetUrl: String) {

    class HttpStatusException(val code: Int, val reason: String) : IOException("HTTP $code $reason")
    class HlsDetectedException(val url: String) : IOException("HLS playlist")

    private class Upstream(val socket: Socket, val input: InputStream, val contentType: String?)

    @Volatile var lastError: String? = null
    @Volatile var stationName: String? = null
    @Volatile var hlsUrl: String? = null
    @Volatile private var closed = false
    private var server: ServerSocket? = null
    private val lock = Any()
    private val live = ArrayList<Socket>()

    fun start(): Int {
        val ss = ServerSocket(0, 4, InetAddress.getByName("127.0.0.1"))
        server = ss
        val t = Thread({ acceptLoop(ss) }, "relay-accept")
        t.isDaemon = true
        t.start()
        return ss.localPort
    }

    fun close() {
        closed = true
        try { server?.close() } catch (ignored: Exception) { }
        closeLive()
    }

    private fun track(s: Socket) {
        synchronized(lock) { live.add(s) }
    }

    private fun closeLive() {
        synchronized(lock) {
            for (s in live) {
                try { s.close() } catch (ignored: Exception) { }
            }
            live.clear()
        }
    }

    private fun acceptLoop(ss: ServerSocket) {
        while (!closed) {
            val client: Socket = try { ss.accept() } catch (e: Exception) { break }
            closeLive()          // a new player connection replaces the old one
            track(client)
            val t = Thread({ serve(client) }, "relay-serve")
            t.isDaemon = true
            t.start()
        }
    }

    private fun hostPortOf(url: String): String {
        return try {
            val u = URL(url)
            if (u.port != -1) u.host + ":" + u.port else u.host
        } catch (e: Exception) { url }
    }

    private fun serve(client: Socket) {
        var headersSent = false
        var hp = hostPortOf(targetUrl)
        var upstream: Upstream? = null
        try {
            client.tcpNoDelay = true
            readHeaderBlock(client.getInputStream())      // ignore the player's request
            val cout = client.getOutputStream()
            val up = openUpstream(targetUrl)
            upstream = up
            hp = hostPortOf(targetUrl)
            val ct = up.contentType ?: "audio/mpeg"
            val head = "HTTP/1.1 200 OK\r\nContent-Type: $ct\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
            cout.write(head.toByteArray(Charsets.ISO_8859_1))
            cout.flush()
            headersSent = true
            lastError = null
            val buf = ByteArray(16384)
            while (!closed) {
                val n: Int = try {
                    up.input.read(buf)
                } catch (e: IOException) {
                    if (!closed) lastError = describeNetError(e, hp)
                    break
                }
                if (n < 0) {
                    if (!closed) lastError = "أغلق الخادم الاتصال (انتهى البث أو انقطع)"
                    break
                }
                try {
                    cout.write(buf, 0, n)
                } catch (e: IOException) {
                    break                                  // the player went away
                }
            }
        } catch (e: HlsDetectedException) {
            lastError = "الرابط عبارة عن قائمة HLS — يتم التحويل تلقائياً"
            if (!closed) hlsUrl = e.url
            sendFailure(client, headersSent)
        } catch (e: Exception) {
            if (!closed) {
                lastError = describeNetError(e, hp)
                sendFailure(client, headersSent)
            }
        } finally {
            closeQuiet(client)
            closeQuiet(upstream?.socket)
        }
    }

    private fun closeQuiet(s: Socket?) {
        try { s?.close() } catch (ignored: Exception) { }
    }

    private fun sendFailure(client: Socket, headersSent: Boolean) {
        if (headersSent) return
        try {
            client.getOutputStream().write(
                "HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
                    .toByteArray(Charsets.ISO_8859_1)
            )
        } catch (ignored: Exception) { }
    }

    // ---------- upstream ----------

    private fun connect(u: URL): Socket {
        val https = u.protocol.equals("https", ignoreCase = true)
        val port = if (u.port != -1) u.port else if (https) 443 else 80
        val plain = Socket()
        try {
            plain.connect(InetSocketAddress(u.host, port), 12000)
        } catch (e: Exception) {
            try { plain.close() } catch (ignored: Exception) { }
            throw e
        }
        plain.tcpNoDelay = true
        plain.soTimeout = 20000
        if (!https) return plain
        val ssl = (SSLSocketFactory.getDefault() as SSLSocketFactory)
            .createSocket(plain, u.host, port, true) as SSLSocket
        val params = ssl.sslParameters
        params.endpointIdentificationAlgorithm = "HTTPS"
        ssl.sslParameters = params
        ssl.soTimeout = 20000
        ssl.startHandshake()
        return ssl
    }

    private fun openUpstream(startUrl: String): Upstream {
        var url = startUrl
        var hops = 0
        while (true) {
            if (closed) throw IOException("closed")
            val u = URL(url)
            val sock = connect(u)
            track(sock)
            var keep = false
            try {
                val auth = if (u.userInfo != null)
                    "Authorization: Basic " + Base64.encodeToString(u.userInfo.toByteArray(), Base64.NO_WRAP) + "\r\n"
                else ""
                val hostHdr = u.host + (if (u.port != -1 && u.port != u.defaultPort) ":" + u.port else "")
                val path = if (u.file.isNullOrEmpty()) "/" else u.file
                val req = "GET $path HTTP/1.0\r\nHost: $hostHdr\r\nUser-Agent: VLC/3.0.18\r\n" +
                    "Accept: */*\r\nIcy-MetaData: 0\r\nConnection: close\r\n$auth\r\n"
                val os = sock.getOutputStream()
                os.write(req.toByteArray(Charsets.ISO_8859_1))
                os.flush()

                val inp = BufferedInputStream(sock.getInputStream(), 32768)
                inp.mark(16)
                val sig = ByteArray(5)
                var got = 0
                while (got < 5) {
                    val r = inp.read(sig, got, 5 - got)
                    if (r < 0) break
                    got += r
                }
                inp.reset()
                if (got == 0) throw IOException("أغلق الخادم الاتصال دون إرسال أي بيانات")
                val sigStr = String(sig, 0, got, Charsets.ISO_8859_1)
                if (!(sigStr.startsWith("ICY") || sigStr.startsWith("HTTP"))) {
                    keep = true                       // raw audio without any headers
                    return Upstream(sock, inp, null)
                }

                val parsed = parseHead(readHeaderBlock(inp))
                val code = parsed.first
                val hdr = parsed.third
                if (code in 300..399) {
                    val loc = hdr["location"] ?: throw IOException("إعادة توجيه بدون عنوان (HTTP $code)")
                    url = URI(url).resolve(loc).toString()
                    hops++
                    if (hops > 6) throw IOException("عدد كبير جداً من إعادة التوجيه")
                    continue
                }
                if (code != 200) throw HttpStatusException(code, parsed.second)
                val name = hdr["icy-name"] ?: hdr["ice-name"]
                if (!name.isNullOrBlank()) stationName = name.trim()

                val ct = hdr["content-type"]
                val ctl = (ct ?: "").lowercase(Locale.US)
                val pl = u.path.lowercase(Locale.US)
                val isPlaylist = ctl.contains("scpls") || ctl.contains("pls+xml") || ctl.contains("mpegurl") ||
                    pl.endsWith(".pls") || pl.endsWith(".m3u")
                if (!isPlaylist && ctl.startsWith("text/html")) {
                    throw IOException("الخادم أعاد صفحة ويب بدلاً من البث الصوتي — تحقق من الرابط (قد يلزم اسم المسار الصحيح مثل /live أو ;)")
                }
                if (isPlaylist) {
                    val body = readUpTo(inp, 16384)
                    if (body.contains("#EXT-X-")) throw HlsDetectedException(url)
                    val next = firstPlaylistUrl(body) ?: throw IOException("قائمة التشغيل لا تحتوي على رابط بث")
                    url = URI(url).resolve(next).toString()
                    hops++
                    if (hops > 6) throw IOException("عدد كبير جداً من إعادة التوجيه")
                    continue
                }
                keep = true
                return Upstream(sock, inp, ct)
            } finally {
                if (!keep) closeQuiet(sock)
            }
        }
    }

    private fun readHeaderBlock(input: InputStream): String {
        val sb = StringBuilder()
        var lineLen = 0
        var n = 0
        while (n < 32768) {
            val b = input.read()
            if (b < 0) throw IOException("أُغلق الاتصال قبل وصول ترويسة الرد")
            n++
            if (b == '\n'.code) {
                if (lineLen == 0 && sb.isNotEmpty()) break
                lineLen = 0
                sb.append('\n')
            } else if (b != '\r'.code) {
                lineLen++
                sb.append(b.toChar())
            }
        }
        return sb.toString()
    }

    private fun parseHead(head: String): Triple<Int, String, Map<String, String>> {
        val lines = head.split('\n')
        val parts = lines[0].split(' ', limit = 3)
        val code = parts.getOrNull(1)?.trim()?.toIntOrNull() ?: -1
        val reason = parts.getOrNull(2)?.trim() ?: ""
        val h = HashMap<String, String>()
        for (i in 1 until lines.size) {
            val ln = lines[i]
            val idx = ln.indexOf(':')
            if (idx > 0) h[ln.substring(0, idx).trim().lowercase(Locale.US)] = ln.substring(idx + 1).trim()
        }
        return Triple(code, reason, h)
    }

    private fun readUpTo(input: InputStream, limit: Int): String {
        val bos = java.io.ByteArrayOutputStream()
        val buf = ByteArray(4096)
        try {
            while (bos.size() < limit) {
                val n = input.read(buf)
                if (n < 0) break
                bos.write(buf, 0, n)
            }
        } catch (ignored: IOException) { }
        return String(bos.toByteArray(), Charsets.ISO_8859_1)
    }

    private fun firstPlaylistUrl(body: String): String? {
        for (raw in body.split('\n')) {
            val ln = raw.trim()
            if (ln.isEmpty() || ln.startsWith("#")) continue
            val l = ln.lowercase(Locale.US)
            if (l.startsWith("file") && ln.contains("=")) {
                val v = ln.substringAfter('=').trim()
                if (v.lowercase(Locale.US).startsWith("http")) return v
            } else if (l.startsWith("http")) {
                return ln
            }
        }
        return null
    }
}

// =====================================================================
//  Acoustic sync analysis: onset + level features, normalized cross-correlation
// =====================================================================

class SyncMeasurement(
    val lagSec: Double,        // delay to add to the radio (negative = TV is ahead of the radio)
    val peak: Double,          // normalized cross-correlation at the peak
    val second: Double,        // best competing peak (outside +-1.5 s)
    val z: Double,             // peak height in standard deviations of the other lags
    val overlapFrac: Double,   // share of the listening window that overlapped the radio history
    val halves: Int,           // -1 not checked, 0 halves disagree, 1 halves agree
    val level: Int             // 0 unreliable, 1 low, 2 medium, 3 high
)

object SyncAnalysis {

    private const val HZ = DelayAudioProcessor.ENV_HZ
    private const val INVALID = -2.0

    private class Prefix(x: DoubleArray) {
        val p = DoubleArray(x.size + 1)
        val p2 = DoubleArray(x.size + 1)

        init {
            for (i in x.indices) {
                p[i + 1] = p[i] + x[i]
                p2[i + 1] = p2[i] + x[i] * x[i]
            }
        }
    }

    private class FeatPair(val a: DoubleArray, val b: DoubleArray) {
        val pa = Prefix(a)
        val pb = Prefix(b)
    }

    private fun boxcar(x: DoubleArray, r: Int): DoubleArray {
        val n = x.size
        val ps = DoubleArray(n + 1)
        for (i in 0 until n) ps[i + 1] = ps[i] + x[i]
        val out = DoubleArray(n)
        for (i in 0 until n) {
            val lo = max(0, i - r)
            val hi = min(n - 1, i + r)
            out[i] = (ps[hi + 1] - ps[lo]) / (hi - lo + 1)
        }
        return out
    }

    private fun logSmooth(env: FloatArray): DoubleArray {
        val lg = DoubleArray(env.size)
        for (i in env.indices) lg[i] = ln(env[i].toDouble() + 2e-4)
        return boxcar(lg, 1)
    }

    /** Onset / spike feature: log level minus the mean of the previous 1.5 s, rises only. */
    fun featuresOnset(env: FloatArray): DoubleArray {
        val n = env.size
        val s = logSmooth(env)
        val ps = DoubleArray(n + 1)
        for (i in 0 until n) ps[i + 1] = ps[i] + s[i]
        val onset = DoubleArray(n)
        val w = 75
        val g = 5
        for (i in 0 until n) {
            val hi = i - g
            val lo = max(0, i - w)
            if (hi - lo >= 10) {
                val base = (ps[hi + 1] - ps[lo]) / (hi - lo + 1)
                val d = s[i] - base
                if (d > 0.0) onset[i] = min(d, 3.0)
            }
        }
        return boxcar(onset, 1)
    }

    /** Level feature: log level with the slow (+-1.5 s) trend removed (swells, lulls, roars). */
    fun featuresLevel(env: FloatArray): DoubleArray {
        val n = env.size
        val s = logSmooth(env)
        val m = boxcar(s, 75)
        val hp = DoubleArray(n)
        for (i in 0 until n) hp[i] = s[i] - m[i]
        val sm = boxcar(hp, 1)
        for (i in 0 until n) sm[i] = sm[i].coerceIn(-3.0, 3.0)
        return sm
    }

    /** NCC for integer shifts m in [mMin, mMax]: mic hop j is compared with radio hop j + di - m. */
    private fun curve(
        f: FeatPair, aS: Int, aE: Int, di: Int,
        mMin: Int, mMax: Int, minN: Int
    ): DoubleArray {
        val a = f.a
        val b = f.b
        val out = DoubleArray(mMax - mMin + 1)
        val nb = b.size
        for (m in mMin..mMax) {
            val jLo = max(aS, m - di)
            val jHi = min(aE - 1, nb - 1 + m - di)
            val n = jHi - jLo + 1
            if (n < minN) {
                out[m - mMin] = INVALID
                continue
            }
            val k = di - m
            var sab = 0.0
            for (j in jLo..jHi) sab += a[j] * b[j + k]
            val sa = f.pa.p[jHi + 1] - f.pa.p[jLo]
            val sa2 = f.pa.p2[jHi + 1] - f.pa.p2[jLo]
            val sb = f.pb.p[jHi + k + 1] - f.pb.p[jLo + k]
            val sb2 = f.pb.p2[jHi + k + 1] - f.pb.p2[jLo + k]
            val nn = n.toDouble()
            val cov = sab - sa * sb / nn
            val va = sa2 - sa * sa / nn
            val vb = sb2 - sb * sb / nn
            out[m - mMin] = if (va > 1e-9 && vb > 1e-9) cov / sqrt(va * vb) else 0.0
        }
        return out
    }

    /** Average of the per-feature curves; a shift is valid only if it is valid for every feature. */
    private fun combined(
        fs: List<FeatPair>, aS: Int, aE: Int, di: Int,
        mMin: Int, mMax: Int, minN: Int
    ): DoubleArray {
        val len = mMax - mMin + 1
        val sum = DoubleArray(len)
        val bad = BooleanArray(len)
        for (f in fs) {
            val c = curve(f, aS, aE, di, mMin, mMax, minN)
            for (i in 0 until len) {
                if (c[i] <= -1.5) bad[i] = true else sum[i] += c[i]
            }
        }
        for (i in 0 until len) sum[i] = if (bad[i]) INVALID else sum[i] / fs.size
        return sum
    }

    private fun argmax(c: DoubleArray): Int {
        var bi = -1
        var best = INVALID
        for (i in c.indices) {
            if (c[i] > best) {
                best = c[i]
                bi = i
            }
        }
        return bi
    }

    /**
     * micT0 = wall-clock time (s) of the centre of mic hop 0; radT0 = wall-clock time (s) at which radio
     * hop 0 would be heard with zero delay. Returns the delay to add to the radio so it lines up with the TV.
     */
    fun analyze(
        micEnv: FloatArray, micT0: Double,
        radEnv: FloatArray, radT0: Double,
        negMaxSec: Double, posMaxSec: Double
    ): SyncMeasurement? {
        val fs = listOf(
            FeatPair(featuresOnset(micEnv), featuresOnset(radEnv)),
            FeatPair(featuresLevel(micEnv), featuresLevel(radEnv))
        )
        val na = fs[0].a.size
        val nb = fs[0].b.size
        if (na < 250 || nb < 250) return null
        val dReal = (micT0 - radT0) * HZ
        val di = floor(dReal).toInt()
        val frac = dReal - di
        val mMin = floor(-negMaxSec * HZ - frac).toInt()
        val mMax = ceil(posMaxSec * HZ - frac).toInt()
        val minN = max(250, (0.5 * na).toInt())
        val c = combined(fs, 0, na, di, mMin, mMax, minN)
        val bi = argmax(c)
        if (bi < 0) return null
        val best = c[bi]
        val bestM = mMin + bi

        val excl = (1.5 * HZ).toInt()
        var second = INVALID
        var cnt = 0
        var s1 = 0.0
        var s2 = 0.0
        for (i in c.indices) {
            val v = c[i]
            if (v <= -1.5) continue
            if (abs((mMin + i) - bestM) <= excl) continue
            cnt++
            s1 += v
            s2 += v * v
            if (v > second) second = v
        }
        var z = 0.0
        if (cnt >= 40) {
            val mean = s1 / cnt
            val sd = sqrt(max(1e-12, s2 / cnt - mean * mean))
            z = (best - mean) / sd
        }
        if (second <= -1.5) second = 0.0

        var delta = 0.0
        if (bi > 0 && bi < c.size - 1 && c[bi - 1] > -1.5 && c[bi + 1] > -1.5) {
            val y1 = c[bi - 1]
            val y2 = c[bi]
            val y3 = c[bi + 1]
            val den = y1 - 2.0 * y2 + y3
            if (den < -1e-9) delta = (0.5 * (y1 - y3) / den).coerceIn(-0.5, 0.5)
        }
        val lagSec = (bestM + delta + frac) / HZ

        val jLo = max(0, bestM - di)
        val jHi = min(na - 1, nb - 1 + bestM - di)
        val overlap = (jHi - jLo + 1).toDouble() / na

        // consistency: both halves of the listening window must point to the same delay
        var halves = -1
        if (na >= 800) {
            val mid = na / 2
            val lo = bestM - 3 * HZ
            val hi = bestM + 3 * HZ
            val ca = combined(fs, 0, mid, di, lo, hi, max(150, (0.6 * mid).toInt()))
            val cb = combined(fs, mid, na, di, lo, hi, max(150, (0.6 * (na - mid)).toInt()))
            val ia = argmax(ca)
            val ib = argmax(cb)
            halves = if (ia >= 0 && ib >= 0 && ca[ia] >= 0.12 && cb[ib] >= 0.12 &&
                abs(ia - ib) <= (0.6 * HZ).toInt()
            ) 1 else 0
        }

        // Calibrated on simulated stadium/commentary mixes: z is measured against the other lags,
        // so it already accounts for the window length and the number of lags searched.
        val prom = best - second
        var level = 0
        if (best >= 0.15 && z >= 4.4 && halves != 0) level = 1
        if (best >= 0.15 && z >= 5.2 && prom >= 0.06 && halves != 0) level = 2
        if (best >= 0.15 && z >= 6.2 && prom >= 0.12 && halves == 1) level = 3
        if (halves == -1) level = min(level, 2)
        if (overlap < 0.7) level = min(level, 2)
        return SyncMeasurement(lagSec, best, second, z, overlap, halves, level)
    }
}

// =====================================================================
//  Activity: programmatic Arabic RTL dark UI
// =====================================================================

/** result == null -> message is the error text; otherwise message holds extra notes. */
class MeasureOutcome(val result: SyncMeasurement?, val message: String, val cancelled: Boolean)

class MainActivity : Activity() {

    companion object {
        private const val REQ_MIC = 4711
        private const val DEFAULT_URL = "http://foxfm.xyz:8080/live1"
        private const val MAX_RETRIES = 5
        private const val K_IDLE = 0
        private const val K_BUSY = 1
        private const val K_OK = 2
        private const val K_ERR = 3
        private val DURATIONS = intArrayOf(10, 15, 20, 30, 45, 60)
    }

    private val cBg = Color.parseColor("#0B1120")
    private val cCard = Color.parseColor("#121B2E")
    private val cStroke = Color.parseColor("#1E2A44")
    private val cField = Color.parseColor("#0F1729")
    private val cBtn = Color.parseColor("#1B2740")
    private val cText = Color.parseColor("#E6EDF7")
    private val cMuted = Color.parseColor("#8B9AB5")
    private val cAccent = Color.parseColor("#10B981")
    private val cRed = Color.parseColor("#F87171")
    private val cRedDark = Color.parseColor("#B91C1C")
    private val cAmber = Color.parseColor("#FBBF24")
    private val cOrange = Color.parseColor("#FB923C")
    private val cMint = Color.parseColor("#6EE7B7")

    private val MATCH = ViewGroup.LayoutParams.MATCH_PARENT
    private val WRAP = ViewGroup.LayoutParams.WRAP_CONTENT

    // engine
    private val proc = DelayAudioProcessor()
    private lateinit var syncSink: SyncAudioSink
    private lateinit var player: ExoPlayer
    private var relay: StreamRelay? = null
    private val ui = Handler(Looper.getMainLooper())
    private lateinit var prefs: SharedPreferences

    // state
    private var wantPlaying = false
    private var everReady = false
    private var retries = 0
    private var currentUrl = ""
    private var streamUrl = ""
    private var currentHls = false
    private var retryRunnable: Runnable? = null
    private var delaySec = 0.0
    private var userMuted = false
    private var measuring = false
    private var pendingMeasure = false
    private val cancelFlag = AtomicBoolean(false)
    private var lastResult: SyncMeasurement? = null
    private var durationSec = 20
    private var confirmUntil = 0L

    // views
    private lateinit var tvHost: TextView
    private lateinit var tvState: TextView
    private lateinit var tvMsg: TextView
    private lateinit var tvReadout: TextView
    private lateinit var tvBadge: TextView
    private lateinit var tvResult: TextView
    private lateinit var etUrl: EditText
    private lateinit var btnPlay: TextView
    private lateinit var chipsScroll: HorizontalScrollView
    private lateinit var chipsRow: LinearLayout
    private lateinit var chipsHint: TextView
    private lateinit var btnMeasure: TextView
    private lateinit var durViews: List<TextView>
    private lateinit var progress: ProgressBar
    private lateinit var tvCountdown: TextView
    private lateinit var tvHistory: TextView
    private lateinit var btnSync: TextView
    private lateinit var btnMute: TextView
    private lateinit var btnReset: TextView
    private lateinit var tvDelayBig: TextView
    private lateinit var seek: SeekBar
    private val pills = ArrayList<TextView>()

    private val ticker = object : Runnable {
        override fun run() {
            tickUi()
            ui.postDelayed(this, 250)
        }
    }

    // ------------------------------------------------------------------ lifecycle

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences("footsync", MODE_PRIVATE)
        window.statusBarColor = cBg
        window.navigationBarColor = cBg
        buildPlayer()
        buildUi()
        etUrl.setText(prefs.getString("last_url", DEFAULT_URL))
        val savedDur = prefs.getInt("dur", 20)
        durationSec = if (DURATIONS.contains(savedDur)) savedDur else 20
        refreshDurationChips()
        rebuildChips()
        setDelay(0.0)
        updateMuteButton()
        updatePlayButton()
        setStream(K_IDLE, "جاهز — الصق رابط البث ثم اضغط «تشغيل»")
        refreshReadout()
        updateControls()
        ui.post(ticker)
    }

    override fun onStop() {
        super.onStop()
        if (measuring) cancelFlag.set(true)   // the mic is only reliable while the app is visible
    }

    override fun onDestroy() {
        ui.removeCallbacksAndMessages(null)
        cancelFlag.set(true)
        relay?.close()
        relay = null
        player.release()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_MIC) {
            val granted = grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED
            val wanted = pendingMeasure
            pendingMeasure = false
            if (granted && wanted && !measuring) startMeasureNow()
            else if (!granted) showResultText("بدون إذن الميكروفون لا يمكن فحص التأخير.", cAmber)
        }
    }

    // ------------------------------------------------------------------ small helpers

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density + 0.5f).toInt()

    private fun shape(fill: Int, radiusDp: Int, stroke: Int = 0): GradientDrawable {
        val g = GradientDrawable()
        g.shape = GradientDrawable.RECTANGLE
        g.cornerRadius = dp(radiusDp).toFloat()
        g.setColor(fill)
        if (stroke != 0) g.setStroke(dp(1), stroke)
        return g
    }

    private fun lp(w: Int, h: Int, weight: Float = 0f, s: Int = 0, t: Int = 0, e: Int = 0, b: Int = 0): LinearLayout.LayoutParams {
        val p = LinearLayout.LayoutParams(w, h, weight)
        p.marginStart = dp(s)
        p.marginEnd = dp(e)
        p.topMargin = dp(t)
        p.bottomMargin = dp(b)
        return p
    }

    private fun tv(text: String, sp: Float, color: Int, bold: Boolean = false): TextView {
        val t = TextView(this)
        t.text = text
        t.setTextColor(color)
        t.textSize = sp
        if (bold) t.typeface = Typeface.DEFAULT_BOLD
        return t
    }

    private fun btn(text: String, fill: Int, fg: Int, radius: Int = 14, sp: Float = 15f): TextView {
        val b = tv(text, sp, fg, true)
        b.gravity = Gravity.CENTER
        b.background = shape(fill, radius)
        b.foreground = RippleDrawable(ColorStateList.valueOf(0x33FFFFFF), null, shape(Color.WHITE, radius))
        b.isClickable = true
        b.setPadding(dp(8), dp(13), dp(8), dp(13))
        return b
    }

    private fun card(parent: LinearLayout): LinearLayout {
        val c = LinearLayout(this)
        c.orientation = LinearLayout.VERTICAL
        c.background = shape(cCard, 18, cStroke)
        c.setPadding(dp(14), dp(14), dp(14), dp(14))
        parent.addView(c, lp(MATCH, WRAP, 0f, 0, 0, 0, 12))
        return c
    }

    private fun toast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }

    private fun fmt1(v: Double): String = String.format(Locale.US, "%.1f", v)

    private fun fmt2(v: Double): String = String.format(Locale.US, "%.2f", v)

    private fun hideKeyboard() {
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(etUrl.windowToken, 0)
    }

    private fun setLook(v: View, enabled: Boolean) {
        v.isEnabled = enabled
        v.alpha = if (enabled) 1f else 0.4f
    }

    // ------------------------------------------------------------------ UI

    private fun buildUi() {
        val scroll = ScrollView(this)
        scroll.setBackgroundColor(cBg)
        scroll.layoutDirection = View.LAYOUT_DIRECTION_RTL
        scroll.isFillViewport = true
        val root = LinearLayout(this)
        root.orientation = LinearLayout.VERTICAL
        root.setPadding(dp(14), dp(16), dp(14), dp(28))
        scroll.addView(root, ViewGroup.LayoutParams(MATCH, WRAP))
        setContentView(scroll)

        // ---- 1. status card
        val status = card(root)
        val top = LinearLayout(this)
        top.orientation = LinearLayout.HORIZONTAL
        top.gravity = Gravity.CENTER_VERTICAL
        tvHost = tv("—", 18f, cText, true)
        tvHost.maxLines = 1
        tvHost.ellipsize = TextUtils.TruncateAt.END
        top.addView(tvHost, lp(0, WRAP, 1f, 0, 0, 8, 0))
        tvState = tv("متوقف", 12f, cMuted, true)
        tvState.setPadding(dp(10), dp(4), dp(10), dp(4))
        tvState.background = shape(cField, 10)
        top.addView(tvState, lp(WRAP, WRAP))
        status.addView(top, lp(MATCH, WRAP))
        tvMsg = tv("", 13.5f, cMuted)
        status.addView(tvMsg, lp(MATCH, WRAP, 0f, 0, 6, 0, 0))
        val div = View(this)
        div.setBackgroundColor(cStroke)
        status.addView(div, lp(MATCH, dp(1), 0f, 0, 12, 0, 12))
        val ro = LinearLayout(this)
        ro.orientation = LinearLayout.HORIZONTAL
        ro.gravity = Gravity.CENTER_VERTICAL
        tvReadout = tv("", 15f, cText, true)
        ro.addView(tvReadout, lp(0, WRAP, 1f, 0, 0, 8, 0))
        tvBadge = tv("", 12f, cText, true)
        tvBadge.setPadding(dp(10), dp(4), dp(10), dp(4))
        tvBadge.visibility = View.GONE
        ro.addView(tvBadge, lp(WRAP, WRAP))
        status.addView(ro, lp(MATCH, WRAP))
        tvResult = tv("", 12.5f, cMuted)
        tvResult.visibility = View.GONE
        tvResult.setLineSpacing(0f, 1.15f)
        status.addView(tvResult, lp(MATCH, WRAP, 0f, 0, 8, 0, 0))

        // ---- 2. stream bar
        val sc = card(root)
        val sr = LinearLayout(this)
        sr.orientation = LinearLayout.HORIZONTAL
        sr.gravity = Gravity.CENTER_VERTICAL
        etUrl = EditText(this)
        etUrl.setSingleLine(true)
        etUrl.hint = "رابط البث http://…"
        etUrl.setHintTextColor(cMuted)
        etUrl.setTextColor(cText)
        etUrl.textSize = 14f
        etUrl.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        etUrl.imeOptions = EditorInfo.IME_ACTION_GO
        etUrl.textDirection = View.TEXT_DIRECTION_LTR
        etUrl.gravity = Gravity.LEFT or Gravity.CENTER_VERTICAL
        etUrl.background = shape(cField, 14, cStroke)
        etUrl.setPadding(dp(12), 0, dp(12), 0)
        etUrl.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_GO) {
                onPlayClicked()
                true
            } else false
        }
        sr.addView(etUrl, lp(0, dp(48), 1f, 0, 0, 8, 0))
        btnPlay = btn("تشغيل", cAccent, Color.WHITE)
        btnPlay.setOnClickListener { onPlayClicked() }
        sr.addView(btnPlay, lp(dp(92), dp(48)))
        sc.addView(sr, lp(MATCH, WRAP))
        chipsScroll = HorizontalScrollView(this)
        chipsScroll.isHorizontalScrollBarEnabled = false
        chipsRow = LinearLayout(this)
        chipsRow.orientation = LinearLayout.HORIZONTAL
        chipsScroll.addView(chipsRow, ViewGroup.LayoutParams(WRAP, WRAP))
        sc.addView(chipsScroll, lp(MATCH, WRAP, 0f, 0, 10, 0, 0))
        chipsHint = tv("اضغط مطولاً على رابط محفوظ لحذفه", 11f, cMuted)
        sc.addView(chipsHint, lp(MATCH, WRAP, 0f, 0, 6, 0, 0))

        // ---- 3. measure card
        val mc = card(root)
        btnMeasure = btn("فحص التأخير", cAccent, Color.WHITE, 16, 20f)
        btnMeasure.setPadding(dp(8), dp(18), dp(8), dp(18))
        btnMeasure.setOnClickListener { if (measuring) cancelFlag.set(true) else beginMeasure() }
        mc.addView(btnMeasure, lp(MATCH, WRAP))
        val dr = LinearLayout(this)
        dr.orientation = LinearLayout.HORIZONTAL
        val dv = ArrayList<TextView>()
        for (d in DURATIONS) {
            val c = tv("$d ث", 14f, cText, true)
            c.gravity = Gravity.CENTER
            c.textDirection = View.TEXT_DIRECTION_LTR
            c.setPadding(0, dp(9), 0, dp(9))
            c.setOnClickListener { selectDuration(d) }
            dr.addView(c, lp(0, WRAP, 1f, 3, 0, 3, 0))
            dv.add(c)
        }
        durViews = dv
        mc.addView(dr, lp(MATCH, WRAP, 0f, 0, 12, 0, 0))
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal)
        progress.max = 1000
        progress.progress = 0
        progress.progressTintList = ColorStateList.valueOf(cAccent)
        progress.progressBackgroundTintList = ColorStateList.valueOf(cStroke)
        mc.addView(progress, lp(MATCH, dp(8), 0f, 0, 12, 0, 0))
        tvCountdown = tv("", 13f, cMuted)
        tvCountdown.visibility = View.GONE
        mc.addView(tvCountdown, lp(MATCH, WRAP, 0f, 0, 8, 0, 0))
        tvHistory = tv("", 12f, cMuted)
        mc.addView(tvHistory, lp(MATCH, WRAP, 0f, 0, 6, 0, 0))

        // ---- 4. action row
        val ar = LinearLayout(this)
        ar.orientation = LinearLayout.HORIZONTAL
        btnSync = btn("زامن الآن", cAccent, Color.WHITE, 14, 14f)
        btnMute = btn("صوت القناة", cBtn, cText, 14, 14f)
        btnReset = btn("إيقاف المزامنة", cBtn, cText, 14, 14f)
        btnSync.setOnClickListener { onSyncClicked() }
        btnMute.setOnClickListener { toggleMute() }
        btnReset.setOnClickListener {
            setDelay(0.0)
            toast("أُعيد تأخير الصوت إلى 0.00 ث")
        }
        ar.addView(btnSync, lp(0, WRAP, 1f, 0, 0, 4, 0))
        ar.addView(btnMute, lp(0, WRAP, 1f, 2, 0, 2, 0))
        ar.addView(btnReset, lp(0, WRAP, 1f, 4, 0, 0, 0))
        root.addView(ar, lp(MATCH, WRAP, 0f, 0, 0, 0, 12))

        // ---- 5. manual fine tuning
        val fc = card(root)
        tvDelayBig = tv("0.00 ث", 46f, Color.WHITE, true)
        tvDelayBig.gravity = Gravity.CENTER
        tvDelayBig.textDirection = View.TEXT_DIRECTION_LTR
        fc.addView(tvDelayBig, lp(MATCH, WRAP))
        seek = SeekBar(this)
        seek.max = 12000
        seek.layoutDirection = View.LAYOUT_DIRECTION_LTR
        seek.progressTintList = ColorStateList.valueOf(cAccent)
        seek.thumbTintList = ColorStateList.valueOf(cAccent)
        seek.progressBackgroundTintList = ColorStateList.valueOf(cStroke)
        seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                if (fromUser) setDelay(p / 100.0, true)
            }

            override fun onStartTrackingTouch(sb: SeekBar?) {}

            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })
        fc.addView(seek, lp(MATCH, WRAP, 0f, 0, 10, 0, 10))
        val pr = LinearLayout(this)
        pr.orientation = LinearLayout.HORIZONTAL
        pr.layoutDirection = View.LAYOUT_DIRECTION_LTR
        val steps = doubleArrayOf(-10.0, -1.0, -0.25, 0.25, 1.0, 10.0)
        val labels = arrayOf("-10 ث", "-1 ث", "-0.25 ث", "+0.25 ث", "+1 ث", "+10 ث")
        for (i in steps.indices) {
            val step = steps[i]
            val p = tv(labels[i], 12.5f, if (step < 0) Color.parseColor("#FCA5A5") else cMint, true)
            p.gravity = Gravity.CENTER
            p.textDirection = View.TEXT_DIRECTION_LTR
            p.maxLines = 1
            p.setPadding(0, dp(11), 0, dp(11))
            p.background = shape(cField, 20, cStroke)
            p.isClickable = true
            p.setOnClickListener { setDelay(delaySec + step) }
            pr.addView(p, lp(0, WRAP, 1f, 2, 0, 2, 0))
            pills.add(p)
        }
        fc.addView(pr, lp(MATCH, WRAP))
    }

    private fun refreshDurationChips() {
        for (i in DURATIONS.indices) {
            val v = durViews[i]
            val sel = DURATIONS[i] == durationSec
            v.background = shape(if (sel) cAccent else cField, 12, if (sel) cAccent else cStroke)
            v.setTextColor(if (sel) Color.WHITE else cText)
        }
    }

    private fun selectDuration(d: Int) {
        if (measuring) return
        durationSec = d
        prefs.edit().putInt("dur", d).apply()
        refreshDurationChips()
    }

    private fun updatePlayButton() {
        btnPlay.text = if (wantPlaying) "إيقاف" else "تشغيل"
        btnPlay.background = shape(if (wantPlaying) cRedDark else cAccent, 14)
    }

    private fun updateMuteButton() {
        btnMute.text = if (userMuted) "صوت القناة (مكتوم)" else "صوت القناة"
        btnMute.setTextColor(if (userMuted) cRed else cText)
    }

    private fun updateControls() {
        setLook(btnSync, !measuring && lastResult != null)
        setLook(btnMute, !measuring)
        setLook(btnReset, !measuring)
        setLook(seek, !measuring)
        for (p in pills) setLook(p, !measuring)
    }

    private fun setStream(kind: Int, msg: String) {
        var label = "متوقف"
        var col = cMuted
        when (kind) {
            K_BUSY -> {
                label = "جارٍ الاتصال"
                col = cAmber
            }
            K_OK -> {
                label = "يعمل"
                col = cAccent
            }
            K_ERR -> {
                label = "خطأ"
                col = cRed
            }
        }
        tvState.text = label
        tvState.setTextColor(col)
        tvMsg.text = msg
        tvMsg.setTextColor(if (kind == K_ERR) cRed else cMuted)
    }

    private fun showResultText(msg: String, color: Int) {
        tvResult.text = msg
        tvResult.setTextColor(color)
        tvResult.visibility = if (msg.isEmpty()) View.GONE else View.VISIBLE
    }

    private fun refreshReadout() {
        val r = lastResult
        var pic = "—"
        if (r != null) {
            val v = fmt1(abs(r.lagSec))
            pic = if (r.lagSec < 0) "\u202A-$v\u202C" else v
        }
        tvReadout.text = if (r == null) "الصورة — • الصوت ${fmt2(delaySec)} ث"
        else "الصورة $pic ث • الصوت ${fmt2(delaySec)} ث"
        if (r == null) {
            tvBadge.visibility = View.GONE
        } else {
            var label = "غير موثوق"
            var bg = "#450A0A"
            var fg = "#F87171"
            when (r.level) {
                3 -> {
                    label = "ثقة عالية"
                    bg = "#064E3B"
                    fg = "#34D399"
                }
                2 -> {
                    label = "ثقة متوسطة"
                    bg = "#14532D"
                    fg = "#BEF264"
                }
                1 -> {
                    label = "ثقة منخفضة"
                    bg = "#431407"
                    fg = "#FB923C"
                }
            }
            tvBadge.text = label
            tvBadge.setTextColor(Color.parseColor(fg))
            tvBadge.background = shape(Color.parseColor(bg), 10)
            tvBadge.visibility = View.VISIBLE
        }
    }

    private fun setDelay(sec: Double, fromSeek: Boolean = false) {
        delaySec = (Math.round(sec * 100.0) / 100.0).coerceIn(0.0, DelayAudioProcessor.MAX_DELAY_SEC)
        proc.setDelaySeconds(delaySec)
        tvDelayBig.text = fmt2(delaySec) + " ث"
        if (!fromSeek) seek.progress = (delaySec * 100.0).roundToInt()
        refreshReadout()
    }

    // ------------------------------------------------------------------ saved URLs

    private fun loadUrls(): MutableList<String> {
        val raw = prefs.getString("urls", "") ?: ""
        val out = ArrayList<String>()
        for (ln in raw.split('\n')) {
            val t = ln.trim()
            if (t.isNotEmpty()) out.add(t)
        }
        return out
    }

    private fun saveUrls(l: List<String>) {
        prefs.edit().putString("urls", l.joinToString("\n")).apply()
    }

    private fun rememberUrl(u: String) {
        val l = loadUrls()
        l.remove(u)
        l.add(0, u)
        while (l.size > 12) l.removeAt(l.size - 1)
        saveUrls(l)
        prefs.edit().putString("last_url", u).apply()
        rebuildChips()
    }

    private fun rebuildChips() {
        chipsRow.removeAllViews()
        val l = loadUrls()
        for (u in l) {
            val shortUrl = u.removePrefix("http://").removePrefix("https://")
            val label = if (shortUrl.length > 28) shortUrl.substring(0, 27) + "…" else shortUrl
            val c = tv(label, 12.5f, cText)
            c.textDirection = View.TEXT_DIRECTION_LTR
            c.setPadding(dp(12), dp(7), dp(12), dp(7))
            c.background = shape(cField, 16, cStroke)
            c.setOnClickListener {
                etUrl.setText(u)
                etUrl.setSelection(etUrl.text.length)
            }
            c.setOnLongClickListener {
                val nl = loadUrls()
                nl.remove(u)
                saveUrls(nl)
                rebuildChips()
                toast("تم حذف الرابط")
                true
            }
            chipsRow.addView(c, lp(WRAP, WRAP, 0f, 0, 0, 6, 0))
        }
        val vis = if (l.isEmpty()) View.GONE else View.VISIBLE
        chipsScroll.visibility = vis
        chipsHint.visibility = vis
    }

    // ------------------------------------------------------------------ player

    private val playerListener = object : Player.Listener {
        override fun onPlaybackStateChanged(playbackState: Int) {
            if (!wantPlaying) return
            when (playbackState) {
                Player.STATE_READY -> {
                    everReady = true
                    retries = 0
                }
                Player.STATE_ENDED -> handleFailure(relay?.lastError ?: "أنهى الخادم البث")
                else -> {}
            }
        }

        override fun onPlayerError(error: PlaybackException) {
            if (!wantPlaying) return
            if (error.errorCode == PlaybackException.ERROR_CODE_BEHIND_LIVE_WINDOW) {
                player.seekToDefaultPosition()
                player.prepare()
                return
            }
            handleFailure(explainError(error))
        }
    }

    private fun buildPlayer() {
        // DefaultAudioSink + our delay processor, wrapped to learn the exact playout latency.
        val defaultSink = DefaultAudioSink.Builder(this)
            .setAudioProcessors(arrayOf<AudioProcessor>(proc))
            .build()
        syncSink = SyncAudioSink(defaultSink, proc)
        val loadControl = DefaultLoadControl.Builder()
            .setBufferDurationsMs(20000, 50000, 1500, 2500)
            .build()
        player = ExoPlayer.Builder(this, SyncRenderersFactory(this, syncSink))
            .setLoadControl(loadControl)
            .build()
        player.setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(C.USAGE_MEDIA)
                .setContentType(C.AUDIO_CONTENT_TYPE_MUSIC)
                .build(),
            true
        )
        player.setWakeMode(C.WAKE_MODE_NETWORK)
        player.addListener(playerListener)
    }

    private fun httpFactory(): DefaultHttpDataSource.Factory {
        return DefaultHttpDataSource.Factory()
            .setUserAgent("VLC/3.0.18")
            .setAllowCrossProtocolRedirects(true)
            .setConnectTimeoutMs(15000)
            .setReadTimeoutMs(20000)
    }

    private fun isHlsUrl(url: String): Boolean = url.lowercase(Locale.US).contains(".m3u8")

    private fun hostOf(url: String): String {
        return try {
            val u = URL(url)
            if (u.port != -1) u.host + ":" + u.port else u.host
        } catch (e: Exception) {
            url
        }
    }

    private fun keepScreen(on: Boolean) {
        if (on) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun onPlayClicked() {
        if (wantPlaying) stopPlayback("تم إيقاف البث") else startPlayback()
    }

    private fun startPlayback() {
        var url = etUrl.text.toString().trim()
        if (url.isEmpty()) {
            setStream(K_ERR, "أدخل رابط البث أولاً")
            return
        }
        if (!url.contains("://")) url = "http://$url"
        val low = url.lowercase(Locale.US)
        if (!(low.startsWith("http://") || low.startsWith("https://"))) {
            setStream(K_ERR, "الرابط يجب أن يبدأ بـ http:// أو https://")
            return
        }
        etUrl.setText(url)
        hideKeyboard()
        rememberUrl(url)
        if (measuring) cancelFlag.set(true)
        retryRunnable?.let { ui.removeCallbacks(it) }
        wantPlaying = true
        everReady = false
        retries = 0
        currentUrl = url
        lastResult = null
        showResultText("", cMuted)
        refreshReadout()
        proc.requestClear()
        proc.setDelaySeconds(delaySec)
        proc.setMuted(userMuted)
        keepScreen(true)
        updatePlayButton()
        updateControls()
        openStream(url, isHlsUrl(url))
    }

    private fun openStream(url: String, hls: Boolean) {
        relay?.close()
        relay = null
        streamUrl = url
        currentHls = hls
        tvHost.text = hostOf(url)
        setStream(K_BUSY, "جارٍ الاتصال بالخادم…")
        val source: MediaSource
        if (hls) {
            val item = MediaItem.Builder()
                .setUri(url)
                .setLiveConfiguration(
                    MediaItem.LiveConfiguration.Builder()
                        .setMinPlaybackSpeed(1f)
                        .setMaxPlaybackSpeed(1f)
                        .build()
                )
                .build()
            source = HlsMediaSource.Factory(httpFactory()).createMediaSource(item)
        } else {
            val r = StreamRelay(url)
            val port = try {
                r.start()
            } catch (e: Exception) {
                handleFailure("تعذّر تشغيل المرحّل المحلي: ${e.message ?: e.javaClass.simpleName}")
                return
            }
            relay = r
            source = ProgressiveMediaSource.Factory(httpFactory())
                .setLoadErrorHandlingPolicy(DefaultLoadErrorHandlingPolicy(1))
                .createMediaSource(MediaItem.fromUri("http://127.0.0.1:$port/"))
        }
        player.setMediaSource(source)
        player.prepare()
        player.playWhenReady = true
    }

    private fun stopPlayback(msg: String?) {
        wantPlaying = false
        retryRunnable?.let { ui.removeCallbacks(it) }
        retryRunnable = null
        if (measuring) cancelFlag.set(true)
        player.stop()
        player.clearMediaItems()
        relay?.close()
        relay = null
        keepScreen(false)
        updatePlayButton()
        updateControls()
        if (msg != null) setStream(K_IDLE, msg)
    }

    private fun handleFailure(reason: String) {
        val hls = relay?.hlsUrl
        if (wantPlaying && hls != null && !currentHls) {
            setStream(K_BUSY, "الرابط قائمة HLS — جارٍ التحويل تلقائياً…")
            openStream(hls, true)
            return
        }
        if (wantPlaying && everReady && retries < MAX_RETRIES) {
            retries++
            setStream(K_BUSY, "انقطع البث — إعادة المحاولة ($retries/$MAX_RETRIES): $reason")
            player.stop()
            retryRunnable?.let { ui.removeCallbacks(it) }
            val url = streamUrl
            val hlsMode = currentHls
            val rr = Runnable {
                retryRunnable = null
                if (wantPlaying) openStream(url, hlsMode)
            }
            retryRunnable = rr
            ui.postDelayed(rr, 2000)
            return
        }
        stopPlayback(null)
        setStream(K_ERR, reason)
    }

    private fun explainError(e: PlaybackException): String {
        val fromRelay = relay?.lastError
        if (fromRelay != null) return fromRelay
        var t: Throwable? = e
        var depth = 0
        while (t != null && depth < 8) {
            when (t) {
                is HttpDataSource.InvalidResponseCodeException ->
                    return describeNetError(StreamRelay.HttpStatusException(t.responseCode, t.responseMessage ?: ""), hostOf(streamUrl))
                is UnknownHostException, is ConnectException, is SocketTimeoutException, is SSLException ->
                    return describeNetError(t, hostOf(streamUrl))
                else -> {}
            }
            t = t.cause
            depth++
        }
        return when (e.errorCode) {
            PlaybackException.ERROR_CODE_IO_NETWORK_CONNECTION_FAILED ->
                "فشل الاتصال بالشبكة — تحقق من الإنترنت ومن الرابط"
            PlaybackException.ERROR_CODE_IO_NETWORK_CONNECTION_TIMEOUT ->
                "انتهت مهلة الاتصال بالخادم"
            PlaybackException.ERROR_CODE_IO_BAD_HTTP_STATUS ->
                "الخادم رفض الطلب (رد HTTP غير ناجح)"
            PlaybackException.ERROR_CODE_IO_CLEARTEXT_NOT_PERMITTED ->
                "الاتصال غير المشفّر (http) غير مسموح على هذا الجهاز"
            PlaybackException.ERROR_CODE_PARSING_CONTAINER_UNSUPPORTED,
            PlaybackException.ERROR_CODE_PARSING_CONTAINER_MALFORMED ->
                "تنسيق الصوت غير معروف أو تالف — تأكد أن الرابط بث صوتي (MP3 / AAC / HLS)"
            PlaybackException.ERROR_CODE_DECODER_INIT_FAILED,
            PlaybackException.ERROR_CODE_DECODING_FORMAT_UNSUPPORTED,
            PlaybackException.ERROR_CODE_DECODING_FORMAT_EXCEEDS_CAPABILITIES ->
                "الجهاز لا يدعم فك ترميز هذا الصوت"
            PlaybackException.ERROR_CODE_DECODING_FAILED ->
                "فشل فك ترميز الصوت"
            PlaybackException.ERROR_CODE_AUDIO_TRACK_INIT_FAILED,
            PlaybackException.ERROR_CODE_AUDIO_TRACK_WRITE_FAILED ->
                "تعذّر فتح مخرج الصوت في الجهاز"
            else -> "${e.errorCodeName}: ${e.message ?: ""}"
        }
    }

    // ------------------------------------------------------------------ periodic UI refresh

    private fun tickUi() {
        val st = player.playbackState
        if (wantPlaying) {
            val r = relay
            val name = r?.stationName
            tvHost.text = if (!name.isNullOrBlank()) name else hostOf(streamUrl)
            if (st == Player.STATE_READY && player.playWhenReady) {
                val filled = proc.historySeconds()
                if (delaySec > filled + 0.5) {
                    setStream(K_OK, "يعمل — جارٍ ملء مخزن التأخير (${fmt1(filled)} من ${fmt1(delaySec)} ث)")
                } else {
                    setStream(K_OK, "يعمل مباشرة")
                }
            } else if (st == Player.STATE_BUFFERING || st == Player.STATE_IDLE) {
                val e = r?.lastError
                if (e != null && retryRunnable == null) setStream(K_BUSY, "جارٍ المحاولة… $e")
            }
        }
        if (wantPlaying && st == Player.STATE_READY) {
            tvHistory.text = "سجل البث المتاح للمقارنة: ${proc.historySeconds().toInt()} ث " +
                "(يجب أن يتجاوز التأخير المتوقع + مدة الفحص)"
        } else {
            tvHistory.text = ""
        }
    }

    // ------------------------------------------------------------------ mute / sync

    private fun toggleMute() {
        if (measuring) return
        userMuted = !userMuted
        proc.setMuted(userMuted)
        updateMuteButton()
    }

    /** Step two of the two-step flow: the delay is applied only here, on the user's tap. */
    private fun onSyncClicked() {
        val r = lastResult
        if (r == null) {
            toast("افحص التأخير أولاً")
            return
        }
        val now = System.currentTimeMillis()
        if (r.level == 0 && now > confirmUntil) {
            confirmUntil = now + 6000
            toast("النتيجة غير موثوقة — اضغط «زامن الآن» مرة أخرى للتطبيق رغم ذلك")
            return
        }
        confirmUntil = 0L
        if (r.lagSec >= 0.0) {
            setDelay(min(r.lagSec, DelayAudioProcessor.MAX_DELAY_SEC))
            showResultText(
                "تم تطبيق تأخير الصوت ${fmt2(delaySec)} ث. للتأكد أعد الفحص: يجب أن يطابق «الصوت» قيمة «الصورة».",
                cMint
            )
        } else {
            setDelay(0.0)
            showResultText(
                "التلفزيون متقدّم على الراديو بمقدار ${fmt2(-r.lagSec)} ث، ولا يمكن تقديم الصوت — تأخير الصوت 0.00 ث.",
                cAmber
            )
        }
    }

    // ------------------------------------------------------------------ measurement

    private fun beginMeasure() {
        if (!wantPlaying || player.playbackState != Player.STATE_READY || proc.currentSampleRate() <= 0) {
            showResultText("شغّل البث وانتظر حتى يبدأ الصوت، ثم افحص التأخير.", cAmber)
            return
        }
        if (proc.ringUnavailable()) {
            showResultText("ذاكرة الجهاز لا تكفي لمخزن التأخير (125 ث).", cRed)
            return
        }
        val hist = proc.historySeconds()
        if (hist < 12.0) {
            showResultText("سجل البث ${hist.toInt()} ث فقط — انتظر حتى يتجاوز 12 ث ثم أعد الفحص.", cAmber)
            return
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            pendingMeasure = true
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_MIC)
            return
        }
        startMeasureNow()
    }

    private fun startMeasureNow() {
        measuring = true
        cancelFlag.set(false)
        proc.setMuted(true)                       // the mic must hear the TV only
        showResultText("", cMuted)
        btnMeasure.text = "إلغاء الفحص"
        btnMeasure.background = shape(cRedDark, 16)
        progress.max = durationSec * 10
        progress.progress = 0
        tvCountdown.visibility = View.VISIBLE
        tvCountdown.text = "الراديو مكتوم مؤقتاً — جارٍ الاستماع للتلفزيون…"
        updateControls()
        val dur = durationSec
        val epoch0 = proc.epoch()
        Thread({ runMeasure(dur, epoch0) }, "measure").start()
    }

    private fun onMeasureTick(elapsed: Double, dur: Int) {
        progress.max = dur * 10
        progress.progress = min(dur * 10, (elapsed * 10).toInt())
        val left = max(0, ceil(dur - elapsed).toInt())
        tvCountdown.text = "الراديو مكتوم مؤقتاً — جارٍ الاستماع للتلفزيون… متبقي $left ث"
    }

    private fun onMeasureAnalyzing() {
        progress.progress = progress.max
        tvCountdown.text = "جارٍ تحليل التطابق بين التلفزيون والراديو…"
    }

    private fun runMeasure(dur: Int, epoch0: Int) {
        val outcome: MeasureOutcome = try {
            doMeasure(dur, epoch0)
        } catch (e: SecurityException) {
            MeasureOutcome(null, "لا يوجد إذن للميكروفون", false)
        } catch (e: Exception) {
            MeasureOutcome(null, "تعذّر الفحص: ${e.javaClass.simpleName} ${e.message ?: ""}", false)
        }
        runOnUiThread { finishMeasure(outcome) }
    }

    private fun createRecord(sr: Int, bufBytes: Int): AudioRecord? {
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val rawOk = am.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true"
        val sources = if (rawOk) intArrayOf(MediaRecorder.AudioSource.UNPROCESSED, MediaRecorder.AudioSource.MIC)
        else intArrayOf(MediaRecorder.AudioSource.MIC)
        for (src in sources) {
            try {
                val r = AudioRecord(src, sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufBytes)
                if (r.state == AudioRecord.STATE_INITIALIZED) return r
                r.release()
            } catch (e: Exception) {
                // try the next source
            }
        }
        return null
    }

    private fun median(l: List<Double>): Double {
        val s = l.sorted()
        return s[s.size / 2]
    }

    private fun spread(l: List<Double>): Double {
        val s = l.sorted()
        val lo = s[(s.size * 0.1).toInt()]
        val hi = s[min(s.size - 1, (s.size * 0.9).toInt())]
        return hi - lo
    }

    /** Background thread: records the TV, then compares it with the pre-delay radio envelope. */
    private fun doMeasure(dur: Int, epoch0: Int): MeasureOutcome {
        val sr = 16000
        val envHz = DelayAudioProcessor.ENV_HZ
        val hop = sr / envHz
        val minBuf = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (minBuf <= 0) return MeasureOutcome(null, "الميكروفون غير مدعوم على هذا الجهاز", false)
        val rec = createRecord(sr, max(minBuf * 4, sr * 2))
            ?: return MeasureOutcome(null, "تعذّر فتح الميكروفون — قد يكون مستخدماً من تطبيق آخر", false)
        try {
            val chunk = ShortArray(hop)
            val hops = FloatArray((dur + 4) * envHz)
            var nHops = 0
            var acc = 0.0
            var inHop = 0
            var prev = 0.0
            var total = 0L
            var minOff = Double.MAX_VALUE
            val m0s = ArrayList<Double>()
            val r0s = ArrayList<Double>()
            val rfb = ArrayList<Double>()
            val ts = AudioTimestamp()
            var lastPoll = 0.0
            var lastUi = 0L

            rec.startRecording()
            if (rec.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                return MeasureOutcome(null, "تعذّر بدء التسجيل من الميكروفون", false)
            }
            val t0 = System.nanoTime()
            while (true) {
                if (cancelFlag.get()) return MeasureOutcome(null, "أُلغي الفحص", true)
                val elapsed = (System.nanoTime() - t0) / 1e9
                if (elapsed >= dur) break
                val n = rec.read(chunk, 0, hop)
                if (n < 0) return MeasureOutcome(null, "خطأ في قراءة الميكروفون (رمز $n)", false)
                val tr = System.nanoTime() / 1e9
                for (k in 0 until n) {
                    val x = chunk[k].toDouble()
                    acc += abs(x - prev)              // first difference, same pre-emphasis as the radio tap
                    prev = x
                    inHop++
                    if (inHop == hop) {
                        if (nHops < hops.size) {
                            hops[nHops] = (acc / hop / 32768.0).toFloat()
                            nHops++
                        }
                        acc = 0.0
                        inHop = 0
                    }
                }
                total += n
                val off = tr - total.toDouble() / sr
                if (off < minOff) minOff = off
                if (tr - lastPoll >= 0.2) {
                    lastPoll = tr
                    if (rec.getTimestamp(ts, AudioTimestamp.TIMEBASE_MONOTONIC) == AudioRecord.SUCCESS) {
                        m0s.add(ts.nanoTime / 1e9 - ts.framePosition.toDouble() / sr)
                    }
                    val rs = proc.currentSampleRate()
                    if (rs > 0) {
                        val r0 = syncSink.r0Seconds(rs)
                        if (!r0.isNaN()) r0s.add(r0)
                        rfb.add(tr - proc.framesQueued().toDouble() / rs)
                    }
                }
                val nowMs = System.currentTimeMillis()
                if (nowMs - lastUi > 100) {
                    lastUi = nowMs
                    val el = elapsed
                    runOnUiThread { onMeasureTick(el, dur) }
                }
            }
            runOnUiThread { onMeasureAnalyzing() }

            // ---- gather both sides on one wall-clock axis
            val skip = (1.2 * envHz).toInt()          // mute ramp + audio already queued in the AudioTrack
            if (nHops - skip < 300) return MeasureOutcome(null, "التسجيل قصير جداً — اختر مدة أطول", false)
            val srRad = proc.currentSampleRate()
            val snap = proc.snapshot(DelayAudioProcessor.ENV_CAP.toDouble() / envHz)
            if (srRad <= 0 || snap == null || snap.epoch != epoch0 || proc.epoch() != epoch0) {
                return MeasureOutcome(null, "تغيّر البث أثناء الفحص — أعد الفحص", false)
            }
            val usable = hops.copyOfRange(skip, nHops)
            val sortedMic = usable.sortedArray()
            val micMedian = sortedMic[sortedMic.size / 2].toDouble()
            if (micMedian < 0.0003) {
                return MeasureOutcome(
                    null,
                    "الميكروفون لا يلتقط صوتاً يُذكر — ارفع صوت التلفزيون وقرّب الهاتف منه، وتأكد من إذن الميكروفون",
                    false
                )
            }
            val quiet = micMedian < 0.0015

            val m0Fallback = minOff - 0.025
            var m0 = m0Fallback
            if (m0s.size >= 3) {
                val mt = median(m0s)
                if (abs(mt - m0Fallback) <= 0.3) m0 = mt
            }
            var usedFallback = false
            var r0Spread = 0.0
            val r0: Double
            if (r0s.size >= 3) {
                r0 = median(r0s)
                r0Spread = spread(r0s)
            } else {
                if (rfb.isEmpty()) return MeasureOutcome(null, "لا توجد بيانات توقيت من مشغّل الصوت — أعد الفحص", false)
                usedFallback = true
                r0 = median(rfb) + 0.30
            }

            val micT0 = m0 + (skip + 0.5) / envHz
            val radT0 = r0 + (snap.firstHop + 0.5) / envHz
            val res = SyncAnalysis.analyze(
                usable, micT0, snap.env, radT0, 30.0, DelayAudioProcessor.MAX_DELAY_SEC
            ) ?: return MeasureOutcome(
                null,
                "بيانات غير كافية للمقارنة — اترك البث يعمل مدة أطول ثم أعد الفحص",
                false
            )

            var level = res.level
            val notes = StringBuilder()
            if (quiet) {
                level = min(level, 1)
                notes.append("صوت التلفزيون خافت عند الميكروفون. ")
            }
            if (usedFallback) {
                level = min(level, 2)
                notes.append("تعذّر قياس زمن مخرج الصوت بدقة (تقدير ±0.3 ث). ")
            }
            if (r0Spread > 0.12) {
                level = min(level, 2)
                notes.append("استقرار التشغيل لم يكن جيداً أثناء الفحص. ")
            }
            val adj = if (level == res.level) res
            else SyncMeasurement(res.lagSec, res.peak, res.second, res.z, res.overlapFrac, res.halves, level)
            return MeasureOutcome(adj, notes.toString().trim(), false)
        } finally {
            try {
                rec.stop()
            } catch (e: Exception) {
                // never started
            }
            rec.release()
        }
    }

    private fun finishMeasure(o: MeasureOutcome) {
        measuring = false
        proc.setMuted(userMuted)                  // restore the sound as it was before the test
        btnMeasure.text = "فحص التأخير"
        btnMeasure.background = shape(cAccent, 16)
        tvCountdown.visibility = View.GONE
        progress.progress = 0
        val r = o.result
        if (o.cancelled) {
            showResultText("أُلغي الفحص.", cMuted)
        } else if (r == null) {
            lastResult = null
            showResultText(o.message, cAmber)
        } else {
            lastResult = r
            val col = when (r.level) {
                3 -> cAccent
                2 -> cMint
                1 -> cAmber
                else -> cOrange
            }
            showResultText(describe(r, o.message), col)
        }
        refreshReadout()
        updateControls()
    }

    private fun describe(r: SyncMeasurement, notes: String): String {
        val sb = StringBuilder()
        val lag = r.lagSec
        if (r.level == 0) {
            sb.append("لم يُعثر على تطابق موثوق بين صوت التلفزيون والراديو (أقرب قيمة ${fmt1(lag)} ث غير مؤكدة). ")
            sb.append("ارفع صوت التلفزيون وقرّب الهاتف منه، واترك البث يعمل مدة أطول من التأخير المتوقع + مدة الفحص، ثم أعد الفحص بمدة 30–60 ث.")
        } else if (lag >= 0.0) {
            sb.append("التلفزيون متأخر عن الراديو بمقدار ${fmt2(lag)} ث. اضغط «زامن الآن» لتأخير صوت الراديو بنفس المقدار.")
        } else {
            sb.append("الراديو متأخر عن التلفزيون: التلفزيون متقدّم بمقدار ${fmt2(-lag)} ث. لا يمكن تقديم صوت الراديو، ")
            sb.append(
                if (delaySec > 0.005) "اضغط «زامن الآن» لإعادة التأخير إلى 0.00 ث."
                else "ويبقى تأخير الصوت 0.00 ث."
            )
        }
        if (r.level > 0) sb.append("  (قوة التطابق ${fmt2(r.peak)})")
        if (notes.isNotEmpty()) sb.append("\n").append(notes)
        return sb.toString()
    }
}
