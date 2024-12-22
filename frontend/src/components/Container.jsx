import React from "react";
import ChatInput from "./ChatInput";
import MessageContainer from "./MessageContainer";

const Container = () => {
  return (
    <div className="flex bg-[#DAD9D9] w-[90%] md:w-[70%] sm:w-[80%] mx-auto h-full mb-10 rounded-2xl py-6 md:py-10 md:px-10 mt-2 ">
      <div className="w-full flex flex-col justify-between  bottom-6 left-0 right-0 mx-auto px-6">
        <div className="flex-grow overflow-y-auto">
        <MessageContainer/>
        </div>
        <div className="items-baseline ">
        <ChatInput />
        </div>
      </div>
    </div>
  );
};

export default Container;
