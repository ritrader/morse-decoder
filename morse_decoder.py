import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter, hilbert

# Словарь Морзе
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '-----': '0', '.----': '1', '..---': '2',
    '...--': '3', '....-': '4', '.....': '5', '-....': '6',
    '--...': '7', '---..': '8', '----.': '9'
}

def bandpass_filter(data, fs, lowcut, highcut, order=5):
    """Полосовой фильтр Баттерворта."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def record_audio(duration, fs):
    """Запись звука с микрофона."""
    print(f"Запись звука ({duration} с)...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio[:, 0]

def decode_morse(signal_env, fs):
    """Декодирование огибающей в текст Морзе."""
    threshold = np.mean(signal_env) * 1.5
    bits = signal_env > threshold

    # Разбиваем на серии равных битов и считаем длительности
    runs = []
    current = bits[0]
    length = 0
    for b in bits:
        if b == current:
            length += 1
        else:
            runs.append((current, length))
            current = b
            length = 1
    runs.append((current, length))

    # Определяем единицу времени (dot) — минимальную длительность сигнала
    durations = [length / fs for flag, length in runs if flag]
    dot = min(durations)

    # Строим строку с . и -
    morse_str = ''
    for flag, length in runs:
        t = length / fs
        if flag:
            morse_str += '.' if t < dot * 1.5 else '-'
        else:
            if t < dot * 1.5:
                morse_str += ''      # пауза между элементами
            elif t < dot * 3 * 1.5:
                morse_str += ' '     # пауза между буквами
            else:
                morse_str += '   '   # пауза между словами

    # Переводим Морзе в текст
    decoded = ''
    for word in morse_str.strip().split('   '):
        for char in word.split(' '):
            decoded += MORSE_CODE_DICT.get(char, '?')
        decoded += ' '
    return decoded.strip()

def main():
    fs = 44100      # частота дискретизации
    duration = 10   # длительность записи в секундах
    audio = record_audio(duration, fs)
    filtered = bandpass_filter(audio, fs, 400, 800)
    envelope = np.abs(hilbert(filtered))
    message = decode_morse(envelope, fs)
    print("Расшифрованное сообщение:", message)

if __name__ == "__main__":
    main()
