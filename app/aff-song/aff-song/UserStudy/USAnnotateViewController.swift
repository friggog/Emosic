//
//  USAnnotateViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 10/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import Foundation

class USAnnotateViewController : AffUIViewController {
    
    var emotion: Int?
    var faceImage: UIImage?
    @IBOutlet weak var ActionButton: RoundedUIButton!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var ValenceSlider: UISlider!
    @IBOutlet weak var ArousalSlider: UISlider!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        imageView.image = faceImage
        if (emotion! > 6) {
            ActionButton.titleLabel?.text = "Finish"
        }
    }

    @IBAction func ButtonClicked(_ sender: Any) {
        // TODO save values
        prtin(ValenceSlider.value, ArousalSlider.value)
        if(emotion! > 6){
            self.performSegue(withIdentifier: "usDoneSegue", sender: self)
        }
        else {
            self.performSegue(withIdentifier: "usNextSegue", sender: self)
        }
    }
}
