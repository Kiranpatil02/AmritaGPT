import React, { useState } from "react";
import { FaMicrophone } from "react-icons/fa";
import { IoSend } from "react-icons/io5";

const ChatInput = () => {
  const [text, setText] = useState("");

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
        className={`
          transition-all duration-300 ease-in-out p-4 cursor-pointer hover:opacity-80
          ${text.trim() ? "bg-white rounded-lg" : "bg-[#A4123F] rounded-full"}
        `}
      >
        <div className="
          transition-all duration-300 ease-in-out">
          {text.trim() ? (
              <IoSend size={25} className="text-black transition-transform hover:scale-110"/>
            ) : (
              <FaMicrophone size={25} className="text-white transition-transform hover:scale-110"/>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatInput;
