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
  function handlesubmit(e){
    e.preventDefault()
    if(query.trim()!=""){

      setquestion((prev)=>[...prev,query]);
      setquery("")
    }
  }
  return (
    <>
    <form onSubmit={handlesubmit}>

    <div>
      {question.map((q,index)=>(
        <UserChat key={index} message={q} duration={setdelay} />
      ))}
    </div>

    <div className="flex p-3 items-center justify-between">
    <div className="
          transition-all duration-300 ease-in-out">
          {text.trim() ? (
              <IoSend size={25} className="text-black transition-transform hover:scale-110"/>
            ) : (
              <FaMicrophone size={25} className="text-white transition-transform hover:scale-110"/>
          )}
        </div>
      <input type="text" placeholder="Ask me anything about Amrita Vishwa Vidyapeetham!" value={query} className="w-full mx-6 py-4 px-6 rounded-lg outline-none text-lg" onChange={Inputchange} disabled={delay} />
      <div className="bg-white rounded-xl p-4">
        <button type="submit"  disabled={delay}>
        <IoSend className="hover:cursor-pointer" size={25}/>
        </button>
      </div>
    </div>
    </form>

    </>
  );
};

export default ChatInput;
