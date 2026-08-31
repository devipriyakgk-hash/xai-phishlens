# LIME Local Explanations

## correctly_flagged_phishing (row 1)
True label: Phishing, Predicted: Phishing
- Prefix/Suffix <= 0.00: -0.1406
- Have_At <= 0.00: -0.1225
- iFrame > 0.00: +0.0740
- URL_Depth <= 2.00: -0.0492
- Web_Forwards > 0.00: -0.0418
- Domain_Age <= 0.00: +0.0319

## correctly_flagged_legitimate (row 0)
True label: Legitimate, Predicted: Legitimate
- Prefix/Suffix <= 0.00: -0.1370
- Have_At <= 0.00: -0.1226
- iFrame <= 0.00: -0.0809
- Web_Forwards <= 0.00: +0.0406
- 0.00 < Domain_Age <= 1.00: -0.0340
- 3.00 < URL_Depth <= 4.00: -0.0268

## misclassified_example (row 16)
True label: Phishing, Predicted: Legitimate
- Prefix/Suffix <= 0.00: -0.1383
- Have_At <= 0.00: -0.1179
- Redirection <= 0.00: -0.0639
- iFrame <= 0.00: -0.0634
- Web_Forwards <= 0.00: +0.0531
- Domain_Age <= 0.00: +0.0240