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
        if(emotion! > 6){
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
            for item in data! {
                if item == data!.last {
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
}
