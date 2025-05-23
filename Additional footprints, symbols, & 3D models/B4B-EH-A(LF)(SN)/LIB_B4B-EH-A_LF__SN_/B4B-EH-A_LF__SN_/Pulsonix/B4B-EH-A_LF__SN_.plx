PULSONIX_LIBRARY_ASCII "SamacSys ECAD Model"
//238891/962533/2.50/4/4/Connector

(asciiHeader
	(fileUnits MM)
)
(library Library_1
	(padStyleDef "c165_h110"
		(holeDiam 1.1)
		(padShape (layerNumRef 1) (padShapeType Ellipse)  (shapeWidth 1.65) (shapeHeight 1.65))
		(padShape (layerNumRef 16) (padShapeType RT)  (shapeWidth 1.6) (shapeHeight 1.65))
	)
	(padStyleDef "s165_h110"
		(holeDiam 1.1)
		(padShape (layerNumRef 1) (padShapeType Rect)  (shapeWidth 1.65) (shapeHeight 1.65))
		(padShape (layerNumRef 16) (padShapeType RT)  (shapeWidth 1.6) (shapeHeight 1.65))
	)
	(textStyleDef "Normal"
		(font
			(fontType Stroke)
			(fontFace "Helvetica")
			(fontHeight 1.27)
			(strokeWidth 0.127)
		)
	)
	(patternDef "SHDRV4W111P0_250_1X4_1250X380X" (originalName "SHDRV4W111P0_250_1X4_1250X380X")
		(multiLayer
			(pad (padNum 1) (padStyleRef s165_h110) (pt 0, 0) (rotation 90))
			(pad (padNum 2) (padStyleRef c165_h110) (pt 2.5, 0) (rotation 90))
			(pad (padNum 3) (padStyleRef c165_h110) (pt 5, 0) (rotation 90))
			(pad (padNum 4) (padStyleRef c165_h110) (pt 7.5, 0) (rotation 90))
		)
		(layerContents (layerNumRef 18)
			(attr "RefDes" "RefDes" (pt 3.75, -3.2) (textStyleRef "Normal") (isVisible True))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(line (pt -2.5 -2.2) (pt 10 -2.2) (width 0.001))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(line (pt 10 -2.2) (pt 10 1.6) (width 0.001))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(line (pt 10 1.6) (pt -2.5 1.6) (width 0.001))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(line (pt -2.5 1.6) (pt -2.5 -2.2) (width 0.001))
		)
		(layerContents (layerNumRef 28)
			(line (pt -2.5 -2.2) (pt 10 -2.2) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(line (pt 10 -2.2) (pt 10 1.6) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(line (pt 10 1.6) (pt -2.5 1.6) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(line (pt -2.5 1.6) (pt -2.5 -2.2) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(line (pt -0.825 -0.825) (pt -0.825 0.825) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(line (pt -0.825 0.825) (pt 0.825 0.825) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(line (pt 0.825 0.825) (pt 0.825 -0.825) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(line (pt 0.825 -0.825) (pt -0.825 -0.825) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(arc (pt 2.5, 0) (radius 0.825) (startAngle 0.0) (sweepAngle 0.0) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(arc (pt 2.5, 0) (radius 0.825) (startAngle 180.0) (sweepAngle 180.0) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(arc (pt 5, 0) (radius 0.825) (startAngle 0.0) (sweepAngle 0.0) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(arc (pt 5, 0) (radius 0.825) (startAngle 180.0) (sweepAngle 180.0) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(arc (pt 7.5, 0) (radius 0.825) (startAngle 0.0) (sweepAngle 0.0) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(arc (pt 7.5, 0) (radius 0.825) (startAngle 180.0) (sweepAngle 180.0) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(line (pt -2.1 -1.8) (pt 9.6 -1.8) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(line (pt 9.6 -1.8) (pt 9.6 1.2) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(line (pt 9.6 1.2) (pt -2.1 1.2) (width 0.025))
		)
		(layerContents (layerNumRef 28)
			(line (pt -2.1 1.2) (pt -2.1 -1.8) (width 0.025))
		)
		(layerContents (layerNumRef 18)
			(line (pt -2.5 1.6) (pt 10 1.6) (width 0.2))
		)
		(layerContents (layerNumRef 18)
			(line (pt -2.5 -2.2) (pt 10 -2.2) (width 0.2))
		)
		(layerContents (layerNumRef 18)
			(line (pt -2.5 -2.2) (pt -2.5 1.6) (width 0.2))
		)
		(layerContents (layerNumRef 18)
			(line (pt 10 -2.2) (pt 10 1.6) (width 0.2))
		)
		(layerContents (layerNumRef 18)
			(arc (pt 0, 2.175) (radius 0.125) (startAngle 0.0) (sweepAngle 0.0) (width 0.25))
		)
		(layerContents (layerNumRef 18)
			(arc (pt 0, 2.175) (radius 0.125) (startAngle 180.0) (sweepAngle 180.0) (width 0.25))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(line (pt -2.75 -2.45) (pt 10.25 -2.45) (width 0.05))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(line (pt 10.25 -2.45) (pt 10.25 1.85) (width 0.05))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(line (pt 10.25 1.85) (pt -2.75 1.85) (width 0.05))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(line (pt -2.75 1.85) (pt -2.75 -2.45) (width 0.05))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(arc (pt 0, 0) (radius 0.35) (startAngle 0.0) (sweepAngle 0.0) (width 0.05))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(arc (pt 0, 0) (radius 0.35) (startAngle 180.0) (sweepAngle 180.0) (width 0.05))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(line (pt 0 -0.5) (pt 0 0.5) (width 0.05))
		)
		(layerContents (layerNumRef Courtyard_Top)
			(line (pt -0.5 0) (pt 0.5 0) (width 0.05))
		)
	)
	(symbolDef "B4B-EH-A_LF__SN_" (originalName "B4B-EH-A_LF__SN_")

		(pin (pinNum 1) (pt 0 mils 0 mils) (rotation 0) (pinLength 200 mils) (pinDisplay (dispPinName true)) (pinName (text (pt 230 mils -25 mils) (rotation 0]) (justify "Left") (textStyleRef "Normal"))
		))
		(pin (pinNum 2) (pt 0 mils -100 mils) (rotation 0) (pinLength 200 mils) (pinDisplay (dispPinName true)) (pinName (text (pt 230 mils -125 mils) (rotation 0]) (justify "Left") (textStyleRef "Normal"))
		))
		(pin (pinNum 3) (pt 0 mils -200 mils) (rotation 0) (pinLength 200 mils) (pinDisplay (dispPinName true)) (pinName (text (pt 230 mils -225 mils) (rotation 0]) (justify "Left") (textStyleRef "Normal"))
		))
		(pin (pinNum 4) (pt 0 mils -300 mils) (rotation 0) (pinLength 200 mils) (pinDisplay (dispPinName true)) (pinName (text (pt 230 mils -325 mils) (rotation 0]) (justify "Left") (textStyleRef "Normal"))
		))
		(line (pt 200 mils 100 mils) (pt 600 mils 100 mils) (width 6 mils))
		(line (pt 600 mils 100 mils) (pt 600 mils -400 mils) (width 6 mils))
		(line (pt 600 mils -400 mils) (pt 200 mils -400 mils) (width 6 mils))
		(line (pt 200 mils -400 mils) (pt 200 mils 100 mils) (width 6 mils))
		(attr "RefDes" "RefDes" (pt 650 mils 300 mils) (justify Left) (isVisible True) (textStyleRef "Normal"))
		(attr "Type" "Type" (pt 650 mils 200 mils) (justify Left) (isVisible True) (textStyleRef "Normal"))

	)
	(compDef "B4B-EH-A_LF__SN_" (originalName "B4B-EH-A_LF__SN_") (compHeader (numPins 4) (numParts 1) (refDesPrefix J)
		)
		(compPin "1" (pinName "1") (partNum 1) (symPinNum 1) (gateEq 0) (pinEq 0) (pinType Unknown))
		(compPin "2" (pinName "2") (partNum 1) (symPinNum 2) (gateEq 0) (pinEq 0) (pinType Unknown))
		(compPin "3" (pinName "3") (partNum 1) (symPinNum 3) (gateEq 0) (pinEq 0) (pinType Unknown))
		(compPin "4" (pinName "4") (partNum 1) (symPinNum 4) (gateEq 0) (pinEq 0) (pinType Unknown))
		(attachedSymbol (partNum 1) (altType Normal) (symbolName "B4B-EH-A_LF__SN_"))
		(attachedPattern (patternNum 1) (patternName "SHDRV4W111P0_250_1X4_1250X380X")
			(numPads 4)
			(padPinMap
				(padNum 1) (compPinRef "1")
				(padNum 2) (compPinRef "2")
				(padNum 3) (compPinRef "3")
				(padNum 4) (compPinRef "4")
			)
		)
		(attr "Manufacturer_Name" "JST (JAPAN SOLDERLESS TERMINALS)")
		(attr "Manufacturer_Part_Number" "B4B-EH-A(LF)(SN)")
		(attr "Mouser Part Number" "306-B4BEHALFSN")
		(attr "Mouser Price/Stock" "https://www.mouser.co.uk/ProductDetail/JST-Commercial/B4B-EH-ALFSN?qs=cdbOS8ANM9DAINrtzhP3dA%3D%3D")
		(attr "Arrow Part Number" "B4B-EH-A(LF)(SN)")
		(attr "Arrow Price/Stock" "https://www.arrow.com/en/products/b4b-eh-a-lf-sn/jst-manufacturing?region=europe")
		(attr "Description" "JST (JAPAN SOLDERLESS TERMINALS) - B4B-EH-A(LF)(SN) - CONNECTOR, HEADER, THT, 2.5MM, 4WAY")
		(attr "<Hyperlink>" "https://datasheet.datasheetarchive.com/originals/distributors/Datasheets_SAMA/4db6e353008cd822f9644e9913ed09f5.pdf")
		(attr "<Component Height>" "6")
		(attr "<STEP Filename>" "B4B-EH-A_LF__SN_.stp")
		(attr "<STEP Offsets>" "X=0;Y=0;Z=0")
		(attr "<STEP Rotation>" "X=0;Y=0;Z=0")
	)

)
