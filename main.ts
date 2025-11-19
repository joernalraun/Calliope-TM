// Diese Liste wird von deiner Webseite automatisch überschrieben
let classes = ["Klasse1", "Klasse2", "Klasse3"]

// Hilfsvariable zum Speichern der zuletzt empfangenen Klasse
let lastClass = ""

// Bluetooth UART starten
bluetooth.startUartService()

bluetooth.onUartDataReceived(serial.delimiters(Delimiters.NewLine), function () {
    let received = bluetooth.uartReadUntil(serial.delimiters(Delimiters.NewLine)).trim()

    // Wiederholungen vermeiden
    if (received == lastClass) return
    lastClass = received

    basic.showString(received)

    // Aktion pro Klasse
    handleClass(received)
})

function handleClass(name: string) {
    if (name == "") return

    // Beispielaktionen – werden dynamisch ergänzt
    if (name == "Klasse1") {
        basic.showIcon(IconNames.Heart)
        music.playTone(440, 200)
    }

    if (name == "Klasse2") {
        basic.showIcon(IconNames.Square)
        music.playTone(262, 200)
    }

    if (name == "Klasse3") {
        basic.showIcon(IconNames.Triangle)
        music.playTone(523, 200)
    }

    // Optional: LED-Pinsteuerung
    // pins.digitalWritePin(DigitalPin.P0, 1)
}
