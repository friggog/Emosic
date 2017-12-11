//
//  USStartViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 10/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import Foundation
import MessageUI

class USStartViewController : AffUIViewController, MFMailComposeViewControllerDelegate {
    
    
    @IBAction func StartButtonClicked(_ sender: Any) {
        makeDataFile()
        let runId = UserDefaults.standard.integer(forKey: "RunNumber")
        UserDefaults.standard.set(runId + 1, forKey: "RunNumber")
        performSegue(withIdentifier: "usStartSegue", sender: self)
    }
    
    func makeDataFile() {
        let path = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0] as String
        let url = URL(fileURLWithPath: path).appendingPathComponent("user_study_data.csv")
        let filePath = url.path
        if !FileManager.default.fileExists(atPath: filePath) {
            do {
                try "id, us_emotion, emotion_label, predicted_valence, predicted_arousal, predicted_emotion_label, rating, annotated_valence, annotated_arousal\n".write(toFile: filePath, atomically: false, encoding: .utf8)
            }
            catch {
                print("ERROR WRITING FILE")
            }
        }
    }
    
    func sendDataEmail(toAddress url: URL) {
        let mailComposer = MFMailComposeViewController()
        mailComposer.mailComposeDelegate = self
        mailComposer.setSubject("Affectone User Study Data")
        mailComposer.setMessageBody("   ", isHTML: false)
        mailComposer.setToRecipients(["cth40@cam.ac.uk"])
        let fileData = try! Data.init(contentsOf: url)
        mailComposer.addAttachmentData(fileData, mimeType: "text/csv", fileName: "Data.csv")
        self.present(mailComposer, animated: true, completion: nil)
    }
    
    @IBAction func MoreButtonPress(_ sender: Any) {
        let path = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0] as String
        let url = URL(fileURLWithPath: path).appendingPathComponent("user_study_data.csv")
        let filePath = url.path
        
        let alert = UIAlertController(title: "User Study Data", message: nil, preferredStyle: UIAlertControllerStyle.alert)
        alert.addAction(UIAlertAction(title: "Cancel", style: UIAlertActionStyle.cancel, handler: nil))
        alert.addAction(UIAlertAction(title: "Email", style: UIAlertActionStyle.default, handler: {action in
            self.sendDataEmail(toAddress: url)
        }))
        alert.addAction(UIAlertAction(title: "Delete", style: UIAlertActionStyle.destructive, handler: {action in
            do {
                try FileManager.default.removeItem(atPath: filePath)
                self.makeDataFile()
            }
            catch {
                print("ERROR DELETING")
            }
        }))
        self.present(alert, animated: true, completion: nil)
    }
    
    @IBAction func cancelButtonClicked(_ sender: Any) {
        navigationController?.dismiss(animated: true, completion: nil)
    }
    
    func mailComposeController(_ controller: MFMailComposeViewController, didFinishWith result: MFMailComposeResult, error: Error?) {
        controller.dismiss(animated: true, completion: nil)
    }
}
