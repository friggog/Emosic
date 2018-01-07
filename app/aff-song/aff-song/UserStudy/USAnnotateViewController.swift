//
//  USAnnotateViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 10/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import Foundation

class USAnnotateViewController : AffUIViewController {
    
    var emotion : Int?
    var faceImage : UIImage?
    var data : [String]?

    @IBOutlet weak var ActionButton: RoundedUIButton!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var ValenceSlider: UISlider!
    @IBOutlet weak var ArousalSlider: UISlider!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        imageView.image = faceImage!
        if (emotion! > 6) {
            ActionButton.titleLabel?.text = "Finish"
        }
    }

    @IBAction func ButtonClicked(_ sender: Any) {
        data!.append("\(ValenceSlider.value)")
        data!.append("\(ArousalSlider.value)")
        saveData()
        if(emotion! == 9){
            self.performSegue(withIdentifier: "usDoneSegue", sender: self)
        }
        else {
            self.performSegue(withIdentifier: "usNextSegue", sender: self)
        }
    }
    
    func saveData() {
        let path = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0] as String
        let url = URL(fileURLWithPath: path).appendingPathComponent("user_study_data.csv")
        if let fileHandle = try? FileHandle(forWritingTo: url) {
            fileHandle.seekToEndOfFile()
            for i in 0..<data!.count {
                let item = data![i]
                if i == data!.count - 1 {
                    fileHandle.write((item).data(using: .utf8)!)
                }
                else {
                    fileHandle.write((item + ", ").data(using: .utf8)!)
                }
            }
            fileHandle.write("\n".data(using: .utf8)!)
            fileHandle.closeFile()
        }
        else {
            print("ERROR SAVING DATA")
        }
    }
    
    @IBAction func cancelButtonClicked(_ sender: Any) {
        let alert = UIAlertController(title: "Exit User Study", message: "Are you sure you would like to exit the study?", preferredStyle: UIAlertControllerStyle.alert)
        alert.addAction(UIAlertAction(title: "Exit", style: UIAlertActionStyle.destructive, handler: { action in
            self.navigationController?.dismiss(animated: true, completion: nil)
        }))
        alert.addAction(UIAlertAction(title: "Cancel", style: UIAlertActionStyle.cancel, handler: nil))
        self.present(alert, animated: true, completion: nil)
    }
}
