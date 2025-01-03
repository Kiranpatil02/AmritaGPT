import React, { useState, useRef } from "react";
import { FaMicrophone } from "react-icons/fa";
import { IoSend } from "react-icons/io5";

const ChatInput = ({ addMessage }) => {
  const [text, setText] = useState("");
  const [isrecording, setisrecording] = useState(false);
  const [audiorecord,setaudiorecord] = useState(null);
  const audioarr = useRef([]);


const MicrophoneFunc = async () =>{
  console.log("clicked")
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  audioarr.current = [];

  recorder.ondataavailable = (event) => {
    if(event.data.size > 0){
      audioarr.current.push(event.data);
    }
  };

  recorder.onstop = () => {
    const audiofile = new Blob(audioarr.current, { type: "audio/webm" });
    sendAudio(audiofile);
    audioarr.current = [];
  };

  recorder.start();
      setaudiorecord(recorder);
      setisrecording(true);
}

const stopRecording = () => {
  if(audiorecord){
    audiorecord.stop();
    setisrecording(false);
  }
};

const sendAudio = async (audioBlob) => {
  const Data = new FormData();
  Data.append("audio", audioBlob, "audio.webm");

  console.log(Data)
  const response = await fetch("http://127.0.0.1:8000/upload-audio", { 
      method: "POST",
      body: Data,
    });

    if(response.ok){
      console.log("Audio sent successfully");
      const transcription = response.headers.get('Transcription');
      const responseText = response.headers.get('Response-Text');
      console.log(transcription, responseText);
      if(transcription){
          addMessage({ user: true, text: transcription });
        }
      if(responseText){
          addMessage({ user: false, text: responseText });
        }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      // Play audio
      await audio.play();
    } else {
      console.error("Not sent...");
    }
};

const handleClick = () => {
  if(isrecording) {
    stopRecording();
  } else {
    MicrophoneFunc();
  }
};


  return (
    <div className="flex p-3 items-center">
      <textarea
        placeholder="Ask me anything about Amrita Vishwa Vidyapeetham!"
        className="w-full mr-4 py-4 px-6 rounded-lg outline-none text-sm md:text-lg resize-none"
        rows="2"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div
        className={`transition-all duration-300 ease-in-out p-4 cursor-pointer hover:opacity-80 ${text.trim() ? "bg-white rounded-lg" : "bg-[#A4123F] rounded-full"}`}
      >
        <div className="transition-all duration-300 ease-in-out">
          {text.trim() ? (
            <IoSend size={25} className="text-black transition-transform hover:scale-110" />
          ) : (
            <FaMicrophone size={25} className={`text-white transition-transform hover:scale-110 ${isrecording ? "animate-pulse animate-bounce" : ""}`} onClick={handleClick} />
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatInput;
