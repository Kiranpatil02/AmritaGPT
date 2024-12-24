import React, { useState } from "react";
import { FaMicrophone } from "react-icons/fa";
import { IoSend } from "react-icons/io5";
import UserChat from "./UserChat";

const ChatInput = () => {
  const [query,setquery]=useState("");
  const [question,setquestion]=useState([]);
  const [delay,setdelay]=useState(false);


  function Inputchange(e){
    setquery(e.target.value);
  }
  function handlesubmit(){
    if(query.trim()!=""){

      setquestion((prev)=>[...prev,query]);
      setquery("")
    }
  }
  return (
    <>
    <div>
      {question.map((q,index)=>(
        <UserChat key={index} message={q} duration={setdelay} />
      ))}
    </div>

    <div className="flex p-3 items-center justify-between">
      <div className="bg-[#A4123F] p-4 rounded-full">
        <FaMicrophone size={25} color="white"/>
      </div>
      <input type="text" placeholder="Ask me anything about Amrita Vishwa Vidyapeetham!" value={query} className="w-full mx-6 py-4 px-6 rounded-lg outline-none text-lg" onChange={Inputchange} disabled={delay} />
      <div className="bg-white rounded-xl p-4">
        <button onClick={handlesubmit} disabled={delay}>
        <IoSend className="hover:cursor-pointer" size={25}/>
        </button>
      </div>
    </div>
    </>
  );
};

export default ChatInput;
